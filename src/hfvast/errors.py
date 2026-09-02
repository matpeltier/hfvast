"""Typed error hierarchy with friendly, actionable messages.

Every user-facing error must say what happened, why, and what to try next.
Secrets must never travel inside these messages.
"""


class HfvastError(Exception):
    """Base class for all hfvast errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ModelInputError(HfvastError):
    """The model argument could not be parsed."""


class ModelNotFoundError(HfvastError):
    """The repository does not exist, is private, or the token lacks access."""


class GatedAccessError(HfvastError):
    """The repository is gated and the token is missing/unauthorized."""


class ModelNotSupportedError(HfvastError):
    """The model cannot be deployed automatically (no resources created)."""

    def __init__(
        self,
        reason: str,
        detected: str = "",
        possible_backend: str = "",
    ) -> None:
        self.reason = reason
        self.detected = detected
        self.possible_backend = possible_backend
        lines = ["This model cannot currently be deployed automatically.", "", f"Reason: {reason}"]
        if detected:
            lines += ["", f"Detected: {detected}"]
        if possible_backend:
            lines += ["", f"Possible manual backend: {possible_backend}"]
        lines += ["", "No cloud resources were created."]
        super().__init__("\n".join(lines))


class PlanError(HfvastError):
    """Planning failed (e.g. variant not found, impossible requirements)."""


class NoCompatibleOfferError(HfvastError):
    """No provider offer satisfied the constraints / cost cap."""

    def __init__(self, cap_hourly: float | None, cheapest_hourly: float | None) -> None:
        self.cap_hourly = cap_hourly
        self.cheapest_hourly = cheapest_hourly
        lines = ["No compatible offer within your limits."]
        if cheapest_hourly is not None:
            lines.append(f"Cheapest compatible offer: ${cheapest_hourly:.2f}/h")
        if cap_hourly is not None:
            if cheapest_hourly:
                lines.append(f"Use --max-hourly-cost {cheapest_hourly + 0.01:.2f} (or higher) to allow it.")
            else:
                lines.append(f"Current cap: ${cap_hourly:.2f}/h")
        lines.append("No Vast resources were created.")
        super().__init__("\n".join(lines))


class CostCapExceededError(HfvastError):
    """The plan exceeds a configured cost cap."""

    def __init__(self, kind: str, estimated: float, cap: float) -> None:
        self.kind = kind
        self.estimated = estimated
        self.cap = cap
        super().__init__(
            f"Deployment aborted: estimated {kind} ${estimated:.2f} exceeds configured cap ${cap:.2f}.\n"
            "No Vast resources were created."
        )


class ProviderError(HfvastError):
    """A cloud provider API call failed."""


class HFHubError(ProviderError):
    """A Hugging Face Hub API call failed."""


class ProviderAuthError(ProviderError):
    """Authentication with the provider failed (bad/missing key)."""


class RateLimitError(ProviderError):
    """Provider rate limit exceeded after retries."""


class GGUFHeaderError(HfvastError):
    """A remote GGUF header could not be fetched or parsed."""
