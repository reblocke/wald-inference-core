# Privacy

## Runtime posture

`wald-inference` is a local numerical Python library. Its public calculation functions accept
ordinary Python values and NumPy arrays and return in-memory domain objects, arrays, and scalar
values.

The package does not:

- make network requests;
- transmit inputs or outputs;
- place values in URLs;
- write inputs to files, databases, browser storage, cookies, or telemetry;
- create user accounts or identifiers;
- log input values by default; or
- provide a server, browser application, or hosted computation service.

NumPy and SciPy are numerical dependencies. Installing the package or resolving dependencies may
contact the configured package index, but calculations do not require network access.

## Data boundary

No clinical data or protected health information is required or expected. Examples and tests use
synthetic numerical inputs or generated baseline fixtures. Applications that call this library own
their input collection, logging, persistence, export, and transmission behavior and must document
and test those boundaries independently.

Public issues, pull requests, code-review excerpts, screenshots, workflow logs, release evidence,
and new fixtures must also use synthetic values and must not contain protected health information,
patient-level data, credentials, unpublished restricted data, or identifying local-path details.

Do not add telemetry, persistence, logging of user values, remote APIs, or server-side processing
without explicit approval and a new privacy review.

## Scientific and clinical boundary

The library computes mathematical quantities under documented Wald assumptions. It does not provide
clinical decision support, determine meaningful clinical thresholds, or validate whether an input
study is appropriate for patient-level use.
