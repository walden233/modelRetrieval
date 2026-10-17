# video_p5

Describe ONE manipulation video from chronologically sampled frames. Use only
observed task actions, interacting objects, necessary targets, and observed results.
Omit unsupported actions and outcomes. When an action cannot be identified,
describe the visible task object or state without guessing an operation.
Return only the requested JSON. All representations refer to the same execution.
Do not include explanations, justifications, uncertainty notes, visibility reports,
confidence, frame references, or dataset/sample/camera identifiers.

## Human focus

Focus on the primary human demonstrator's manipulation and hand-object interactions.
Ignore bystanders and unrelated activity, including recording, keyboard use, and
preparation unrelated to the manipulation. Mention another actor only when that
actor visibly contributes to the task. Preserve action ownership.

## Robot focus

Focus on the robot arm/end effector, manipulated objects, and observed effects.
Ignore unrelated human presence and activity. Include human participation only
for an observed handover, object positioning, or cooperation. Preserve action
ownership. Describe an approach or contact as such; do not invent a grasp.

## Output instructions

Write natural-language values in English. Produce ALL SIX representations together
from these SAME frames in this ONE response. S2 uses description plus labels;
S0, S1, S3, S4 and S5 have their own fields. Do not copy the longer description
into the concise baseline. Keep facts consistent across representations, while
preserving each representation's level of detail. Output these task-content fields only:

0. baseline_description (S0): One concise English task sentence with the main
   action, manipulated object, and necessary target. Omit ALL incidental colors,
   appearance, background, and clothing. Identify an object by its category or
   functional role. Preserve task-defining distinctions without color words.

1. description: A concise natural account, normally 1–3 sentences, of relevant
   objects, main actions, and observed results. Useful task-object appearance is
   allowed. Omit background, incidental movement, and comments on observability.
2. labels: capability_tags, task_complexity, environment_tags, scene_category.
   Use the supplied values only. Capability labels may be empty; unknown is valid
   for a categorical label. Do not force an unlisted action into a wrong label.
   Complexity describes visible steps, not robot difficulty. Do not infer an
   industrial scene from the robot alone. Other fields use open-vocabulary verbs.
3. object_action_text: Directly write separated "Objects:" and "Operations:"
   sections and an optional "Result:" section. Preserve the binding of operations
   to objects and source/target roles. Ignore incidental colors, patterns, brands,
   decoration, and style. Keep a distinction only when necessary for the task.
4. action_focus_text: Directly write "Objects:" and "Operations:" sections and
   an optional "Result:" section. Emphasize actions, order, and functional object
   roles. Generalize incidental appearance, material, and object category without
   losing the operation's meaning. Ignore incidental colors, patterns, brands,
   decoration, and style. Retain only task-essential distinctions.
5. structured: Objects, ordered operations, and result as semantic JSON. Each object
   has local id, name, role. Each operation has action, object (local ID or null),
   actor (primary/other/joint/unknown), and target (null, {"object_id":"o2"}, or
   {"location":"task-relevant location"}). Result is null if not established.

Omit an unestablished result consistently from all text fields and use null in
structured.result. Never append why it is omitted. Do not describe nonexistent
actions in negative sentences. S3/S4 are direct outputs, not reconstruction rules.

MANDATORY APPEARANCE RULE FOR S3/S4:
By default omit ALL color words, including accurate colors of task objects.
A uniquely visible colored button is simply "the button"; its color does not
identify a different task. The only exceptions are distinguishing two otherwise
identical interacting objects, or a color change that is itself the task.
Never include background/cloth patterns or clothing colors. In S4 also omit
incidental material and style. Return task content only, without explaining any
inclusion, omission, or interpretation.
