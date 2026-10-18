# scene_p5

Write a concise English description of the shared manipulation in one scene's
independent human/robot observations. Inputs are task texts or semantic JSON from
ONE selected S group, not videos; tags are not extra observed actions. Retain
operations supported on both sides, necessary functional object roles, order,
and shared observed outcomes. Ignore camera, background, actor appearance, and
incidental colors, materials, brands, or style. Mention actor types only for actual
human-robot interaction. Omit nonshared phases and contradictory details; a missing
phase alone does not imply a different operation.

Return only JSON with description, or null for description if no reliable shared
manipulation is identified. No explanations, reasons, disagreements, uncertainty
or visibility notes, confidence, or source citations.
