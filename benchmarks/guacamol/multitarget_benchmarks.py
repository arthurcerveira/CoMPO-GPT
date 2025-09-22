from rdkit import Chem

from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.scoring_function import GeometricMeanScoringFunction
from guacamol.utils.descriptors import qed

from guacamol.common_scoring_functions import (
    TargetResponseScoringFunction,
    CNS_MPO_ScoringFunction,
    SyntheticAccessibilityScoringFunction,
    RdkitScoringFunction,
    BBBResponseScoringFunction
)

# Import sascorer from RDKit contrib
import os
import sys
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))

import sascorer


def alzheimer_mpo_benchmark() -> GoalDirectedBenchmark:
    """
    Benchmark to evaluate multi-target molecules active against Alzheimer's disease.
    Targets considered:
    - Acetylcholinesterase (AChE)
    - Monoamine oxidase B (MAO-B)
    Other criteria:
    - Pass through blood-brain barrier (BBB)
    - Physicochemical Properties for Optimal Brain Exposure
    - Synthetical accessibility
    """
    ache_scorer = TargetResponseScoringFunction(target='AChE')
    maob_scorer = TargetResponseScoringFunction(target='MAOB')

    mean_effectiveness = GeometricMeanScoringFunction(
        [ache_scorer, maob_scorer]
    )

    bbb_scorer = BBBResponseScoringFunction()

    # Physicochemical Properties for Optimal Brain Exposure
    cnsm_mpo = CNS_MPO_ScoringFunction()

    # Synthetical accessibility
    synthetic_accessibility = SyntheticAccessibilityScoringFunction(
        sascorer.calculateScore
    )

    mean_scorer = GeometricMeanScoringFunction(
        [mean_effectiveness, bbb_scorer, cnsm_mpo, synthetic_accessibility]
    )

    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(
        name='Alzheimer MPO',
        objective=mean_scorer,
        contribution_specification=specification
    )


def schizophrenia_mpo_benchmark() -> GoalDirectedBenchmark:
    """
    Benchmark to evaluate multi-target molecules active against schizophrenia.
    Targets considered:
    - Dopamine D2 receptor (D2)
    - 5-hydroxytryptamine receptor 2A (5-HT2A)
    Other criteria:
    - Pass through blood-brain barrier (BBB)
    - Physicochemical Properties for Optimal Brain Exposure
    - Synthetical accessibility
    """
    d2_scorer = TargetResponseScoringFunction(target='D2R')
    _5ht2a_scorer = TargetResponseScoringFunction(target='_5HT2A')
    mean_effectiveness = GeometricMeanScoringFunction(
        [d2_scorer, _5ht2a_scorer]
    )

    bbb_scorer = BBBResponseScoringFunction()

    # Physicochemical Properties for Optimal Brain Exposure
    cnsm_mpo = CNS_MPO_ScoringFunction()

    # Synthetical accessibility
    synthetic_accessibility = SyntheticAccessibilityScoringFunction(
        sascorer.calculateScore
    )

    mean_scorer = GeometricMeanScoringFunction(
        [mean_effectiveness, bbb_scorer, cnsm_mpo, synthetic_accessibility]
    )

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name='Schizophrenia MPO',
        objective=mean_scorer,
        contribution_specification=specification
    )


def parkinson_mpo_benchmark() -> GoalDirectedBenchmark:
    """
    Benchmark to evaluate multi-target molecules active against parkinson.
    Targets considered:
    - Dopamine D2, and D3 receptor (D2R, D3R)
    Other criteria:
    - Pass through blood-brain barrier (BBB)
    - Physicochemical Properties for Optimal Brain Exposure
    - Synthetical accessibility
    """
    d2_scorer = TargetResponseScoringFunction(target='D2R')
    d3_scorer = TargetResponseScoringFunction(target='D3R')
    mean_effectiveness = GeometricMeanScoringFunction(
        [d2_scorer, d3_scorer]
    )

    bbb_scorer = BBBResponseScoringFunction()

    # Physicochemical Properties for Optimal Brain Exposure
    cnsm_mpo = CNS_MPO_ScoringFunction()

    # Synthetical accessibility
    synthetic_accessibility = SyntheticAccessibilityScoringFunction(
        sascorer.calculateScore
    )

    mean_scorer = GeometricMeanScoringFunction(
        [mean_effectiveness, bbb_scorer, cnsm_mpo, synthetic_accessibility]
    )

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(
        name='Parkinson MPO',
        objective=mean_scorer,
        contribution_specification=specification
    )
