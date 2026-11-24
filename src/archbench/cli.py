"""Thin CLI over the application modules."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .adapters.registry import AdapterRegistry, default_registry
from .domain.common import content_hash
from .domain.execution import AttemptStatus
from .execution.controller import Controller
from .execution.qualification import qualify_configuration
from .planning.compiler import compile_plan, load_experiment, load_plan, persist_plan
from .planning.loader import load_scenario
from .planning.provenance import source_revision
from .reporting.report import publish_report
from .secrets import SecretsError, load_local_secrets
from .storage.artifacts import ArtifactStore
from .storage.database import Database
from .storage.reconcile import reconcile


class Workspace:
    def __init__(self, project: Path, results: Path, secrets: Path | None = None) -> None:
        self.project = project.resolve()
        self.results = results if results.is_absolute() else self.project / results
        self.secrets = load_local_secrets(self.project, secrets)
        self.db = Database(self.results / "index.sqlite")
        self.store = ArtifactStore(self.results)

    def registry(self) -> AdapterRegistry:
        return default_registry(self.secrets)

    def code_revision(self) -> str:
        return source_revision(self.project).label


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def cmd_qualify(ws: Workspace, args: argparse.Namespace) -> int:
    loaded = load_experiment(Path(args.experiment), ws.project)
    scenario, _ = load_scenario(ws.project / loaded.spec.scenario, ws.project)
    registry = ws.registry()
    failed = 0
    for config, frozen in loaded.configurations:
        record = qualify_configuration(ws.db, registry, config, frozen, scenario, ws.code_revision())
        report = record.payload["report"]
        failed += not report["passed"]
        print(f"{config.key}: {'QUALIFIED' if report['passed'] else 'NOT QUALIFIED'} ({record.record_id})")
        for check in report["checks"]:
            print(f"  [{'ok' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    return 1 if failed else 0


def cmd_plan(ws: Workspace, args: argparse.Namespace) -> int:
    plan = compile_plan(Path(args.experiment), base=ws.project, db=ws.db)
    persist_plan(ws.db, ws.store, plan)
    print(f"froze plan for {plan.experiment_id}: sha256 {plan.plan_sha256}")
    if plan.source.dirty:
        print(f"warning: source tree is dirty ({plan.source.label}); config bytes are preserved in the plan")
    return 0


def cmd_run(ws: Workspace, args: argparse.Namespace) -> int:
    plan = load_plan(ws.db, args.experiment_id)
    summary = Controller(ws.db, ws.store, ws.registry()).run(plan)
    _print(summary.model_dump(mode="json"))
    return 0


def cmd_status(ws: Workspace, args: argparse.Namespace) -> int:
    tasks = ws.db.list_tasks(args.experiment_id)
    for task in tasks:
        reason = f"  ({task.status_reason})" if task.status_reason else ""
        print(f"{task.task_id}  {task.status.value}{reason}")
    print(f"{len(ws.db.list_responses(args.experiment_id))}/{len(tasks)} tasks have responses")
    return 0


def cmd_reconcile(ws: Workspace, args: argparse.Namespace) -> int:
    with ws.store.lock(args.experiment_id):
        report = reconcile(ws.db, ws.store, args.experiment_id)
    _print(report.model_dump(mode="json"))
    return 0 if report.integrity_ok else 1


def cmd_evaluate(ws: Workspace, args: argparse.Namespace) -> int:
    from .evaluation.runner import EvaluationPreflightError, resume_evaluation, start_evaluation

    registry = ws.registry()
    if args.resume:
        run_id = args.resume
    else:
        if not args.policy:
            print("either --policy or --resume is required", file=sys.stderr)
            return 2
        try:
            run_id = start_evaluation(
                ws.db, ws.store, args.experiment_id, Path(args.policy), base=ws.project, registry=registry,
                code_revision=ws.code_revision(),
            ).record_id
        except EvaluationPreflightError as exc:
            print(f"evaluation not started: {exc}", file=sys.stderr)
            return 2
    status = resume_evaluation(ws.db, ws.store, registry, run_id)
    _print(status.model_dump(mode="json"))
    return 0


def cmd_evaluation_status(ws: Workspace, args: argparse.Namespace) -> int:
    from .evaluation.runner import evaluation_status

    _print(evaluation_status(ws.db, args.evaluation_run_id).model_dump(mode="json"))
    return 0


def cmd_evaluate_v2(ws: Workspace, args: argparse.Namespace) -> int:
    from .evaluation.selection_runner import resume_selection_evaluation, selection_status, start_selection_evaluation
    registry = ws.registry()
    try:
        run_id = args.resume
        if run_id:
            run = ws.db.get_record(run_id)
            if run is None or run.experiment_id != args.experiment_id:
                raise ValueError("resume run does not belong to the requested experiment")
        else:
            run_id = start_selection_evaluation(ws.db, ws.store, registry, args.experiment_id,
                Path(args.policy), base=ws.project, code_revision=ws.code_revision()).record_id
        report = resume_selection_evaluation(ws.db, ws.store, registry, run_id)
    except (ValueError, RuntimeError) as exc:
        print(f"v2 evaluation not completed: {exc}", file=sys.stderr)
        return 2
    status = selection_status(ws.db, run_id)
    _print({**status, "report_id": report.record_id,
            "artifacts": {name: ref.relpath for name, ref in ws.db.artifacts_for(report.record_id).items()}})
    return 0 if status["all_required_outputs_valid"] else 1


def cmd_selection_status(ws: Workspace, args: argparse.Namespace) -> int:
    from .evaluation.selection_runner import selection_status
    _print(selection_status(ws.db, args.evaluation_run_id))
    return 0


def cmd_review(ws: Workspace, args: argparse.Namespace) -> int:
    import webbrowser

    from .evaluation.review_app import ReviewApp, serve

    app = ReviewApp(ws.db, ws.store, args.evaluation_run_id, args.reviewer)
    server, url = serve(app, args.port)
    queue = app.queue()
    print(f"checking {queue['total']} response(s) of {args.evaluation_run_id} as {args.reviewer}: "
          f"{queue['done']} done, {queue['total'] - queue['done']} to go")
    print(f"open {url}  (Ctrl-C to stop; every decision is saved as you make it)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    queue = app.queue()
    print(f"\nstopped: {queue['done']}/{queue['total']} checked, {queue['counts']['flagged']} flagged")
    return 0


def cmd_review_export(ws: Workspace, args: argparse.Namespace) -> int:
    from .evaluation.review import export_reviews
    from .evaluation.runner import load_run

    ctx = load_run(ws.db, args.evaluation_run_id)
    written = export_reviews(ws.db, ws.store, ctx.run, Path(args.out))
    print(f"exported {len(written) // 2} flagged response(s) for full review to {args.out}")
    return 0


def cmd_review_import(ws: Workspace, args: argparse.Namespace) -> int:
    from .evaluation.review import ReviewImportError, import_review
    from .evaluation.runner import load_run

    ctx = load_run(ws.db, args.evaluation_run_id)
    failed = 0
    for name in args.files:
        try:
            record = import_review(ws.db, ws.store, ctx.run, ctx.rubric, Path(name), supersede=args.supersede)
        except ReviewImportError as exc:
            failed += 1
            print(f"{name}: REJECTED: {exc}", file=sys.stderr)
            continue
        superseded = f" (supersedes {record.supersedes})" if record.supersedes else ""
        print(f"{name}: imported as {record.record_id}{superseded}")
    return 1 if failed else 0


def cmd_adjudicate(ws: Workspace, args: argparse.Namespace) -> int:
    from .evaluation.adjudication import adjudicate
    from .evaluation.runner import load_run

    ctx = load_run(ws.db, args.evaluation_run_id)
    _print(adjudicate(ws.db, ctx.run, ctx.policy, ctx.rubric).model_dump(mode="json"))
    return 0


def cmd_report(ws: Workspace, args: argparse.Namespace) -> int:
    record = publish_report(ws.db, ws.store, args.experiment_id, args.evaluation, code_revision=ws.code_revision())
    path = ws.results / args.experiment_id / "reports" / record.record_id / "report.md"
    print(f"report revision {record.record_id}: {path}")
    return 0


def cmd_cross_report(ws: Workspace, args: argparse.Namespace) -> int:
    from .reporting.cross_model import CrossModelError, publish

    scenarios = (ws.project / args.scenarios) if args.scenarios else None
    try:
        data = publish(ws.db, ws.store, args.evaluation_run_ids, Path(args.out), scenarios=scenarios, title=args.title)
    except CrossModelError as exc:
        print(f"cross-model report not written: {exc}", file=sys.stderr)
        return 2
    models = len({c["configuration"] for c in data["cells"]})
    print(f"wrote {args.out}: {models} model(s), {len(data['runs'])} evaluation run(s), data {data['data_sha256'][:16]}")
    return 0


def cmd_inspect(ws: Workspace, args: argparse.Namespace) -> int:
    attempt = ws.db.get_attempt(args.attempt_id)
    task = ws.db.get_task(attempt.task_id)
    artifacts = ws.db.artifacts_for(attempt.attempt_id)
    out: dict[str, object] = {
        "attempt": attempt.model_dump(mode="json"),
        "task": task.model_dump(mode="json"),
        "artifacts": {n: {"relpath": r.relpath, "sha256": r.sha256, "bytes": r.size_bytes} for n, r in artifacts.items()},
    }
    if "metrics.json" in artifacts:
        out["metrics"] = ws.store.read_json(artifacts["metrics.json"])
    if "native-request.json" in artifacts:
        request = ws.store.read_json(artifacts["native-request.json"])
        out["native_request_sha256"] = content_hash(request)
    if "final-answer.txt" in artifacts and attempt.status is AttemptStatus.SUCCEEDED:
        answer = ws.store.read_verified(artifacts["final-answer.txt"]).decode("utf-8")
        out["final_answer_preview"] = answer[:600]
    _print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="ArchDecisionBench harness")
    parser.add_argument("--project", type=Path, default=Path("."), help="project root for relative input paths")
    parser.add_argument("--results", type=Path, default=Path("results"), help="results directory")
    parser.add_argument("--secrets", type=Path, default=None,
                        help="local credentials file (default: <project>/secrets.local.yaml if present)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("qualify", help="qualify every configuration referenced by an experiment")
    p.add_argument("experiment")
    p.set_defaults(func=cmd_qualify)

    p = sub.add_parser("plan", help="compile and freeze an experiment plan")
    p.add_argument("experiment")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="execute or resume a frozen experiment")
    p.add_argument("experiment_id")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="show task status")
    p.add_argument("experiment_id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("reconcile", help="detect interrupted attempts and artifact integrity problems")
    p.add_argument("experiment_id")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("evaluate", help="start or resume automatic evaluation stages (validation to review selection)")
    p.add_argument("experiment_id")
    p.add_argument("--policy")
    p.add_argument("--resume", metavar="EVALUATION_RUN_ID")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("evaluation-status", help="show stage coverage for an evaluation run")
    p.add_argument("evaluation_run_id")
    p.set_defaults(func=cmd_evaluation_status)

    p = sub.add_parser("evaluate-v2", help="Luna extraction, Jev preference labels, sampled/flagged Haiku audits")
    p.add_argument("experiment_id")
    p.add_argument("--policy", default="configs/evaluation/agentic-design-bias-v2.yaml")
    p.add_argument("--resume", metavar="EVALUATION_RUN_ID")
    p.set_defaults(func=cmd_evaluate_v2)

    p = sub.add_parser("selection-status", help="show v2 preference-label evaluation coverage")
    p.add_argument("evaluation_run_id")
    p.set_defaults(func=cmd_selection_status)

    p = sub.add_parser("review", help="open the review app: check every response's judgement, approve or flag it")
    p.add_argument("evaluation_run_id")
    p.add_argument("--reviewer", required=True, help="your reviewer identifier, recorded on every check")
    p.add_argument("--port", type=int, default=0, help="local port (default: any free port)")
    p.add_argument("--no-browser", action="store_true", help="print the URL instead of opening a browser")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("review-export", help="export blinded packets and templates for flagged responses")
    p.add_argument("evaluation_run_id")
    p.add_argument("--out", required=True, help="directory for reviewer files (never overwritten)")
    p.set_defaults(func=cmd_review_export)

    p = sub.add_parser("review-import", help="import completed full human review files")
    p.add_argument("evaluation_run_id")
    p.add_argument("files", nargs="+")
    p.add_argument("--supersede", action="store_true", help="record a correction of the reviewer's prior review")
    p.set_defaults(func=cmd_review_import)

    p = sub.add_parser("adjudicate", help="append adjudication records for checked and reviewed responses")
    p.add_argument("evaluation_run_id")
    p.set_defaults(func=cmd_adjudicate)

    p = sub.add_parser("report", help="publish a report revision from an evaluation run")
    p.add_argument("experiment_id")
    p.add_argument("--evaluation", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("cross-report", help="write a static HTML report comparing models across evaluation runs")
    p.add_argument("evaluation_run_ids", nargs="+", help="one evaluation run per probe and model")
    p.add_argument("--out", required=True, help="HTML file to write")
    p.add_argument("--scenarios", help="probe family directory, to list probes that have no answers yet")
    p.add_argument("--title", default="Agentic design bias")
    p.set_defaults(func=cmd_cross_report)

    p = sub.add_parser("inspect", help="inspect one invocation attempt")
    p.add_argument("attempt_id")
    p.set_defaults(func=cmd_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ws = Workspace(args.project, args.results, args.secrets)
    except SecretsError as exc:
        print(f"unsafe or invalid secrets file: {exc}", file=sys.stderr)
        return 2
    try:
        return int(args.func(ws, args))
    finally:
        ws.db.close()


if __name__ == "__main__":
    raise SystemExit(main())
