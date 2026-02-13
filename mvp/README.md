## CLIP MVP Demo

This demo verifies that image and text can be mapped into a shared semantic space
using a pretrained CLIP model.

Given one image and several candidate captions, we compute cosine similarity
between image and text embeddings.

Result shows that semantically matched caption achieves highest similarity,
which motivates multi-task learning on top of shared vision-language representations.

