"""Method layer, model-agnostic algorithms.

Every submodule takes a :class:`models.base.ModelBackend` (or none) as its
one point of contact with the concrete model. Nothing here imports a concrete
backend, swap the model by changing the ``model.backend`` field in the config.
"""

# WHICH CHAINS ARE THE ANTIBODY. One definition, because the metrics and the verifiers PARTITION THE
# SAME COMPLEX with it: dockq_abag splits antibody from antigen, ipSAE, diversity, the biophysical
# signals and the map builders each do the same split. Separate copies can drift, and a drift does
# not raise, it makes two numbers on one row describe different complexes.
#
# It lives here, in the package root, because this is the only place both method.metrics and
# method.verifier can import without a cycle, metrics already depending on verifier and not the
# reverse. ORDERED, not a set: evaluate.py builds the --antibody-chains default as ",".join(...),
# which a set would emit in arbitrary order.
ANTIBODY_CHAINS = ("H", "L")
