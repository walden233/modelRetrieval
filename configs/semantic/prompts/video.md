# video_p9

Describe ONE manipulation video from chronological frames. Include only observed
actions, interacting objects, necessary targets, and observed results. If an action
is unidentified, describe the visible task object or state without guessing.
Return only the requested JSON, with task content and no explanations, reasons,
uncertainty or visibility notes, confidence, frame references, or dataset/sample/camera IDs.

## Human focus

Focus on the primary human demonstrator's manipulation and hand-object interactions.
Ignore bystanders, recording, keyboard use, and preparation unrelated to the task.
Mention other actors only for visible task contributions; preserve action ownership.

## Robot focus

Focus on the robot arm/end effector, manipulated objects, and observed effects.
Ignore unrelated human presence and activity. Include humans only for observed
handover, object positioning, or cooperation; preserve action ownership.
Describe approach or contact accurately; do not invent a grasp.

## Output instructions

Write all natural-language values in English in one response. Keep actions, object
roles, and outcomes consistent across fields. Put tags only in labels, never in
description or the other texts.

1. description: In 1-3 natural, concise sentences, describe relevant
   objects, main actions, and observed results.  Omit unhelpful
   colors, background, clothing, and incidental movement. Do not speculate.
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
   an optional "Result:" section. Use functional roles instead of concrete object
   names throughout. Keep a concrete category or structural feature only when
   needed to define the operation's mechanism. Preserve object_action_text's
   visible actions, order, source/target relations, and result; do not add steps
   or reduce distinct operations to generic object movement.
5. structured: Objects, ordered operations, and result as semantic JSON. Each object
   has local id, name, role. Each operation has action, object (local ID or null),
   actor (primary/other/joint/unknown), and target (null, {"object_id":"o2"}, or
   {"location":"task-relevant location"}). Result is null if not established.

S3/S4 example (these two fields only; not facts about the current video):
Hypothetical observation: a pen is grasped and lifted out of a pen holder.
```json
{
  "object_action_text": "Objects: a pen, a pen holder.\nOperations: grasp the pen and lift it out of the pen holder.\nResult: the pen is removed from the holder.",
  "action_focus_text": "Objects: an elongated object, a storage container.\nOperations: grasp the object and extract it upward from the container.\nResult: the object is outside the container."
}
```
Use the current frames for facts and return every required field.

Omit unestablished results from all texts and use null in structured.result.
Do not describe absent actions in negative sentences.

MANDATORY APPEARANCE RULE FOR S3/S4:
By default omit ALL color words, including accurate colors of task objects.
A uniquely visible colored button is simply "the button"; its color does not
identify a different task. The only exceptions are distinguishing two otherwise
identical interacting objects, or a color change that is itself the task.
Never include background/cloth patterns or clothing colors. In S4 also omit
incidental material and style. Return task content only, without explaining any
inclusion, omission, or interpretation.
