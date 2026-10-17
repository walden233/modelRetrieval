# scene_p4

Write one concise English description of the shared manipulation in the supplied
independent human/robot observations of one scene. The inputs are semantic task
representations from ONE selected S group, not original videos. They may be
concise descriptions, natural descriptions with tags, separated object/action
text, or semantic JSON. Do not treat tags as additional observed actions. Retain
operations supported across both human and robot observations, necessary
functional object roles, operation order, and
shared observed outcomes. Ignore camera, background, actor appearance, incidental
colors, materials, brands, and style. Mention actor types only for actual human-
robot interaction. Omit nonshared phases and contradictory details. Missing a
phase in one observation does not imply a different operation.

Return only JSON with description. Use null if no reliable shared manipulation
can be identified. Do not output explanations, justifications, disagreements,
uncertainty notes, confidence, visibility reports, or source citations.
