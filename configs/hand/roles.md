Identify the task-relevant human hands in synchronized views of ONE demonstration.
The camera images contain provisional hand boxes and skeletons labeled H0, H1, etc.
These numbers are LOCAL TO EACH CAMERA, can change after occlusion, and are not
anatomical left/right labels. Different cameras may show the same physical hand.
Use the original task context, object contact, tool use and continuous motion.
Do not choose a hand just because it is still, visible longer, or detected clearly.
A hand supporting the manipulated object is a support hand, not an idle hand.
A keyboard-side hand is idle only if its behavior is unrelated to the demonstrated task.
Background false detections are not physical hands. Only assign an identity when
the correspondence is supported; leave ambiguous or unseen fragments unassigned.

Return ONE JSON object, without explanations or markdown:
{
  "task": "one short factual task description",
  "main_hand": "A or null",
  "hands": [{"id": "A", "role": "main"}, {"id": "B", "role": "support"}],
  "cameras": [{"camera": "exact camera ID", "segments": [
    {"track_id": 0, "hand_id": "A", "start_s": 0.0, "end_s": 2.5,
     "confidence": "high"}
  ]}]
}

Use consistent physical hand IDs A, B, ... ACROSS ALL VIEWS AND TRACK FRAGMENTS.
Allowed roles: main (the tool/action-performing hand), support (stabilizing or
cooperatively manipulating the task object), idle (unrelated to the task), unknown.
For a symmetric bimanual task choose one main hand consistently across views and
retain the other as support. There is at most one main hand. Set main_hand to null
if its identity cannot be established; never invent a main hand to fill the schema.
Each camera must appear exactly once. Segments link a visible track to a physical
hand over the stated time interval. Multiple fragments may belong to one hand,
but two simultaneous detections in one camera cannot be that same physical hand.
Use seconds relative to the COMMON scene time origin printed below, not frame
number or video playback time. Main/support segments must cover task engagement,
including purposeful approach and withdrawal, and exclude unrelated preparation.
Use separate intervals if a hand changes activity. Confidence is high, medium or
low; omit ambiguous mappings rather than guessing. A camera may have no segments.
Track ranges and sampled IDs below describe detections, not ground-truth roles.
