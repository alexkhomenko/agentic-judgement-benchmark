# Reliability challenge

The following situations occur in production:

- The Gmail synchronization history has expired, so incremental sync can no longer continue from the last checkpoint.
- The same email notification is delivered more than once.
- A worker crashes immediately after writing to Google Calendar, before recording that the write happened.
- A receipt PDF attachment cannot be parsed.
- An external source API starts rate limiting requests.

Explain whether your recommended architecture remains suitable. For each situation, describe how it is detected and how the system recovers. State any changes you would make to your design.
