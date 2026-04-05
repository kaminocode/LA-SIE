BASE_EXPERIENCES = [
    "SIENoVar",
    "SIE",
    "SIEOnlyEqui",
    "VICReg",
    "SimCLR",
    "VICRegPartInv",
    "SimCLROnlyEqui",
    "SIERotColor",
    "SimCLRAugSelf",
    "SimCLRAugSelfRotColor",
    "SimCLROnlyEquiRotColor",
    "SimCLREquiModRotColor",
    "SimCLREquiMod",
    "VICRegEquiMod",
]

LATENT_ACTION_PAIR_EXPERIENCES = [
    "sie_oracle",
    "direct_full_matrix_2v",
    "direct_skewexp_2v",
    "latentcode_to_full_matrix_2v",
    "sharedgen_fixed_2v",
    "sharedgen_learned_2v",
    "sharedgen_learned_2v_identity",
    "sharedgen_learned_2v_inverse",
]

LATENT_ACTION_TRIPLET_EXPERIENCES = [
    "sharedgen_learned_3v_no_comp",
    "sharedgen_learned_3v_comp",
]

LATENT_ACTION_EXPERIENCES = (
    LATENT_ACTION_PAIR_EXPERIENCES + LATENT_ACTION_TRIPLET_EXPERIENCES
)

ROT_COLOR_EXPERIENCES = [
    "SIERotColor",
    "SimCLRAugSelfRotColor",
    "SimCLROnlyEquiRotColor",
    "SimCLREquiModRotColor",
]

ALL_EXPERIENCES = BASE_EXPERIENCES + LATENT_ACTION_EXPERIENCES


def is_latent_action_experience(experience):
    return experience in LATENT_ACTION_EXPERIENCES


def is_triplet_experience(experience):
    return experience in LATENT_ACTION_TRIPLET_EXPERIENCES


def uses_rotcolor_dataset(experience):
    return experience in ROT_COLOR_EXPERIENCES
