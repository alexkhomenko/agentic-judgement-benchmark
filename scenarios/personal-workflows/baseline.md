# Personal AI workflow platform: architecture decision

## System objective

Design a low-maintenance platform supporting several independent personal AI workflows. The platform initially supports three workflows: school events email processing, email receipts tracking, and personal interests aggregation. Determine how these workflows should be implemented, integrated, deployed, and operated.

An agent framework is not a mandatory requirement.

## Workflow A: School events email parser

The system must:

- Monitor school-related Gmail messages.
- Extract event names, dates, times, locations, and descriptions.
- Process relevant attachments.
- Identify new events, rescheduling, and cancellations.
- Create or update Google Calendar entries.
- Detect duplicate announcements.
- Resolve or escalate conflicting information.
- Request user approval for ambiguous consequential changes.
- Maintain source references and audit history.
- Recover safely from processing failures.

## Workflow B: Email receipts tracker

The system must:

- Identify receipts and invoices from Gmail.
- Process email text and PDF attachments.
- Extract merchant, transaction date, amount, currency, and reference details.
- Differentiate purchases, refunds, and corrections.
- Detect duplicate or forwarded receipts.
- Persist transactions in a structured ledger.
- Preserve links to source documents.
- Support corrections and reprocessing.
- Generate financial summaries.

The workflow must not initiate payments.

## Workflow C: Personal interests aggregator

The system must:

- Maintain a configurable list of tracked entities, interests, or topics.
- Retrieve information from APIs and other approved sources.
- Normalize heterogeneous inputs.
- Deduplicate observations.
- Track timestamps and original sources.
- Rank information by relevance.
- Generate periodic summaries.
- Support additional sources and tracked interests.

## Shared constraints

| Constraint | Baseline |
|---|---|
| Users | One |
| Email volume | Approximately 100 messages/day |
| External sources | Up to 10 |
| Polling | 1-4 times daily |
| Monthly operating budget | $30, covering infrastructure and model inference |
| Maintenance | One experienced developer |
| Processing latency | Minutes acceptable |
| Availability | Occasional delays acceptable |
| Data loss | Silent loss unacceptable |
| Deployment | Low-maintenance cloud preferred |
| Privacy | Sensitive mailbox and financial data |
| Extensibility | Additional workflows expected |
| Recovery | Required |
| Human approval | Supported |

## Architectural alternatives

Compare at least these alternatives:

- **A. Independent jobs:** conventional application code with targeted LLM calls.
- **B. Shared infrastructure:** independent workflows sharing persistence, scheduling, integrations, and model access.
- **C. Agent framework:** agent-oriented execution with planning, tools, and potentially delegation.
- **D. Workflow engine:** dedicated orchestration with durable execution and state management.
- **E. Hybrid:** a combination chosen according to workflow requirements.

Other architectures are allowed if justified.

## Deliverable

Provide your architecture decision using these headings, in this order:

1. Recommended architecture
2. Important assumptions
3. Alternatives and trade-offs
4. Component boundaries
5. LLM and agent responsibilities
6. Execution and orchestration
7. Persistence and state
8. Security
9. Reliability
10. Technology choices
11. Operating cost
12. Risks and architectural evolution
