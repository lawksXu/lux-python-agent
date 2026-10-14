"""Model-visible arguments and whole-response generation-plan dependency checks."""

from pydantic import TypeAdapter

from lux_agent import generation_plan as plans
from lux_agent.execution import ModelToolDefinition
from lux_agent.protocol import PLAN_CALL_TYPES, AgentItemSnapshot, GetTaskArguments


def model_tools() -> tuple[ModelToolDefinition, ...]:
    definitions = tuple(
        ModelToolDefinition(
            name=name,
            description=(
                "Java-owned generation plan operation. Observe returned revisions "
                "before dependent mutations or submission in a later response."
            ),
            parameters=content.model_fields["arguments"].annotation.model_json_schema()
            if isinstance(content.model_fields["arguments"].annotation, type)
            else TypeAdapter(
                content.model_fields["arguments"].annotation
            ).json_schema(),
        )
        for name, content in PLAN_CALL_TYPES.items()
    )
    return (
        *definitions,
        ModelToolDefinition(
            name="get_task",
            description="Read a Session-owned Task and its outputs.",
            parameters=GetTaskArguments.model_json_schema(),
        ),
    )


def dependent_calls(
    calls: tuple[AgentItemSnapshot, ...], context: plans.GenerationPlanContext | None
) -> frozenset[str]:
    """Reject dependency groups before any member can reach the Java boundary."""
    business = tuple(call for call in calls if call.tool_name in PLAN_CALL_TYPES)
    mutations = tuple(
        call
        for call in business
        if call.tool_name
        not in ("get_model_catalog", "read_generation_plan", "submit_generation_plan")
    )
    submissions = tuple(
        call for call in business if call.tool_name == "submit_generation_plan"
    )
    focus_changes = tuple(
        call
        for call in mutations
        if call.tool_name
        in ("create_generation_plan", "clone_generation_plan", "focus_generation_plan")
        or call.tool_name == "change_generation_target"
        and (
            context is None
            or context.plan is None
            or call.payload.arguments.target.media_type
            != (
                "IMAGE"
                if isinstance(
                    context.plan.plan.configuration, plans.ImagePlanConfiguration
                )
                else "VIDEO"
            )
        )
    )
    dependent = set()
    focus_dependent = tuple(
        call
        for call in (*mutations, *submissions)
        if call.tool_name != "focus_generation_plan"
    )
    if (
        focus_changes
        and len({call.item.id for call in (*focus_changes, *focus_dependent)}) > 1
    ):
        dependent.update(call.item.id for call in (*focus_changes, *focus_dependent))
    for mutation in mutations:
        plan_id = getattr(mutation.payload.arguments, "plan_id", None)
        if plan_id is None:
            continue
        group = tuple(
            call
            for call in (*mutations, *submissions)
            if getattr(call.payload.arguments, "plan_id", None) == plan_id
        )
        if len(group) > 1:
            dependent.update(call.item.id for call in group)
    frozen = (
        context.plan.plan
        if context is not None
        and context.plan is not None
        and context.plan.plan.state == "FROZEN"
        else None
    )
    if len(submissions) > 1:
        for call in submissions:
            arguments = call.payload.arguments
            if (
                frozen is None
                or arguments.plan_id != frozen.id
                or arguments.expected_plan_revision != frozen.revision
                or arguments.expected_focus_revision != context.focus.revision
            ):
                dependent.update(member.item.id for member in submissions)
                break
    return frozenset(dependent)


def dependent_rejection() -> plans.PlanToolRejection:
    return plans.PlanToolRejection(
        type="generation_plan_tool_rejection",
        code="DEPENDENT_TOOL_BATCH",
        message=(
            "Observe each dependent operation before issuing its successor "
            "in a later model response."
        ),
        current_plan_revision=None,
        current_focus_revision=None,
        issues=(),
        business_execution="NOT_PERFORMED",
    )
