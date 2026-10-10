"""Pure planning for changes to active search and transaction state."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeAlias

from .tools import SearchCriteria

if TYPE_CHECKING:
    from .controller import SessionState


ImmutableValue: TypeAlias = (
    None | bool | int | float | str
    | tuple["ImmutableValue", ...]
    | Mapping[str, "ImmutableValue"]
)


def _freeze(value: Any) -> ImmutableValue:
    """Copy JSON-like values into recursively immutable value snapshots."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    raise TypeError(f"unsupported mutable booking evidence value: {type(value).__name__}")


@dataclass(frozen=True)
class ConfirmedBookingRecord:
    """Deeply immutable confirmed-booking value suitable for session history."""

    booking_id: str
    hold_id: str
    flight: ImmutableValue
    passenger: str
    simulated: bool
    status: Literal["confirmed"]
    additional_fields: Mapping[str, ImmutableValue]


def _booking_snapshot(booking: Mapping[str, Any]) -> ConfirmedBookingRecord:
    standard = {"booking_id", "hold_id", "flight", "passenger", "simulated", "status"}
    return ConfirmedBookingRecord(
        booking_id=booking["booking_id"],
        hold_id=booking["hold_id"],
        flight=_freeze(booking["flight"]),
        passenger=booking["passenger"],
        simulated=booking["simulated"],
        status="confirmed",
        additional_fields=_freeze({key: value for key, value in booking.items() if key not in standard}),
    )


@dataclass(frozen=True)
class ActiveStateInvalidation:
    """Fields an integration should invalidate in its mutable active context."""

    search: bool = False
    selection: bool = False
    hold: bool = False
    booking_reference: bool = False


@dataclass(frozen=True)
class StateTransitionPlan:
    """A proposed transition, including confirmed records to archive first.

    ``confirmed_records_to_preserve`` contains deeply immutable value
    snapshots suitable for appending to booking history before clearing active
    references. The planner does not mutate state or records.
    """

    criteria_changed: bool
    changed_criteria: tuple[str, ...]
    search_succeeded: bool
    active_invalidation: ActiveStateInvalidation
    confirmed_records_to_preserve: tuple[ConfirmedBookingRecord, ...] = ()

    @property
    def is_noop(self) -> bool:
        return not self.criteria_changed

    @property
    def should_apply(self) -> bool:
        """Whether the proposed transition may be committed atomically."""
        return self.search_succeeded and self.criteria_changed


def _confirmed_record(state: SessionState) -> tuple[ConfirmedBookingRecord, ...]:
    booking = state.booking
    required = {"booking_id", "hold_id", "flight", "passenger", "simulated", "status"}
    if (isinstance(booking, Mapping) and required.issubset(booking)
            and booking.get("status") == "confirmed"):
        return (_booking_snapshot(booking),)
    return ()


def plan_state_transition(
    state: SessionState,
    next_criteria: SearchCriteria,
    *,
    search_succeeded: bool = True,
) -> StateTransitionPlan:
    """Plan a criteria transition after its search outcome is known.

    Failed or timed out searches preserve all current active state so callers
    can roll back atomically. On successful material changes, active search,
    selection, hold, and booking reference are invalidated, while confirmed
    booking evidence is returned for preservation in history.
    """
    current = state.criteria
    changed = tuple(
        name for name in SearchCriteria.__dataclass_fields__
        if getattr(current, name) != getattr(next_criteria, name)
    )
    transition = bool(changed and search_succeeded)
    return StateTransitionPlan(
        criteria_changed=bool(changed),
        changed_criteria=changed,
        search_succeeded=search_succeeded,
        active_invalidation=ActiveStateInvalidation(
            search=transition,
            selection=transition,
            hold=transition,
            booking_reference=transition,
        ),
        confirmed_records_to_preserve=_confirmed_record(state) if transition else (),
    )


def plan_passenger_change(
    state: SessionState, next_passenger: str | None
) -> StateTransitionPlan:
    """Plan clearing active transaction references for a passenger change.

    Past confirmed booking records are surfaced for history preservation and
    never rewritten to reflect the new passenger.
    """
    changed = state.passenger != next_passenger
    return StateTransitionPlan(
        criteria_changed=changed,
        changed_criteria=("passenger",) if changed else (),
        search_succeeded=True,
        active_invalidation=ActiveStateInvalidation(
            selection=changed,
            hold=changed,
            booking_reference=changed,
        ),
        confirmed_records_to_preserve=_confirmed_record(state) if changed else (),
    )
