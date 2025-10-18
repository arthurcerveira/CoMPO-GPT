import datetime
import random
import json
import logging
from collections import OrderedDict
from typing import List, Dict, Any

import guacamol
from guacamol.distribution_learning_benchmark import DistributionLearningBenchmark, DistributionLearningBenchmarkResult
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.benchmark_suites import distribution_learning_benchmark_suite
from guacamol.utils.data import get_time_string
from guacamol.utils.chemistry import is_valid, canonicalize_list

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _SmilesListGenerator(DistributionMatchingGenerator):
    """
    Simple generator that samples SMILES strings from a provided list.
    """

    def __init__(self, smiles: List[str]) -> None:
        self.smiles = smiles

    def generate(self, number_samples=None, only_valid: bool = False, only_unique: bool = False) -> List[str]:
        smiles = self.smiles

        if only_valid:
            smiles = [smi for smi in smiles if is_valid(smi)]

        if only_unique:
            smiles = list(set(canonicalize_list(smiles, include_stereocenters=False)))
        
        if number_samples is None:
            return smiles
        
        return random.sample(smiles, k=number_samples)
    

def assess_distribution_learning(model: DistributionMatchingGenerator,
                                 chembl_training_file: str,
                                 json_output_file='output_distribution_learning.json',
                                 benchmark_version='v1') -> None:
    """
    Assesses a distribution-matching model for de novo molecule design.

    Args:
        model: Model to evaluate
        chembl_training_file: path to ChEMBL training set, necessary for some benchmarks
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
    """
    _assess_distribution_learning(model=model,
                                  chembl_training_file=chembl_training_file,
                                  json_output_file=json_output_file,
                                  benchmark_version=benchmark_version,
                                  number_samples=10000)


def _assess_distribution_learning(model: DistributionMatchingGenerator,
                                  chembl_training_file: str,
                                  json_output_file: str,
                                  benchmark_version: str,
                                  number_samples: int) -> None:
    """
    Internal equivalent to assess_distribution_learning, but allows for a flexible number of samples.
    To call directly only for testing.
    """
    logger.info(f'Benchmarking distribution learning, version {benchmark_version}')
    benchmarks = distribution_learning_benchmark_suite(chembl_file_path=chembl_training_file,
                                                       version_name=benchmark_version,
                                                       number_samples=number_samples)

    results = _evaluate_distribution_learning_benchmarks(model=model, benchmarks=benchmarks)

    benchmark_results: Dict[str, Any] = OrderedDict()
    benchmark_results['guacamol_version'] = guacamol.__version__
    benchmark_results['benchmark_suite_version'] = benchmark_version
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['samples'] = model.generate(100)
    benchmark_results['results'] = [vars(result) for result in results]

    logger.info(f'Save results to file {json_output_file}')
    with open(json_output_file, 'wt') as f:
        f.write(json.dumps(benchmark_results, indent=4))


def _evaluate_distribution_learning_benchmarks(model: DistributionMatchingGenerator,
                                               benchmarks: List[DistributionLearningBenchmark]
                                               ) -> List[DistributionLearningBenchmarkResult]:
    """
    Evaluate a model with the given benchmarks.
    Should not be called directly except for testing purposes.

    Args:
        model: model to assess
        benchmarks: list of benchmarks to evaluate
        json_output_file: Name of the file where to save the results in JSON format
    """

    logger.info(f'Number of benchmarks: {len(benchmarks)}')

    results = []
    for i, benchmark in enumerate(benchmarks, 1):
        logger.info(f'Running benchmark {i}/{len(benchmarks)}: {benchmark.name}')
        result = benchmark.assess_model(model)
        logger.info(f'Results for the benchmark "{result.benchmark_name}":')
        logger.info(f'  Score: {result.score:.6f}')
        logger.info(f'  Sampling time: {str(datetime.timedelta(seconds=int(result.sampling_time)))}')
        logger.info(f'  Metadata: {result.metadata}')
        results.append(result)

    logger.info('Finished execution of the benchmarks')

    return results


def assess_distribution_learning_from_smiles(smiles: List[str],
                                             chembl_training_file: str,
                                             json_output_file: str = None,
                                             benchmark_version: str = 'v1',
                                             number_samples: int = 10000) -> None:
    """
    Assesses distribution-learning benchmarks using a list of SMILES as the sampling source.

    Args:
        smiles: list of SMILES strings to sample from
        chembl_training_file: path to ChEMBL training set, necessary for some benchmarks
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
        number_samples: number of samples each benchmark should request from the generator
    """
    logger.info(f'Benchmarking distribution learning from SMILES, version {benchmark_version}')

    generator = _SmilesListGenerator(smiles)

    benchmarks = distribution_learning_benchmark_suite(
        chembl_file_path=chembl_training_file,
        version_name=benchmark_version,
        number_samples=number_samples,
    )

    results = _evaluate_distribution_learning_benchmarks(model=generator, benchmarks=benchmarks)

    benchmark_results: Dict[str, Any] = OrderedDict()
    benchmark_results['guacamol_version'] = guacamol.__version__
    benchmark_results['benchmark_suite_version'] = benchmark_version
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['samples'] = generator.generate(100)
    benchmark_results['results'] = [vars(result) for result in results]

    if json_output_file is None:
        return benchmark_results

    logger.info(f'Save results to file {json_output_file}')
    with open(json_output_file, 'wt') as f:
        f.write(json.dumps(benchmark_results, indent=4))

