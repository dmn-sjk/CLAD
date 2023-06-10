from clad.classification.cladc_utils import *
import os
from torch.utils.data import ConcatDataset
from avalanche.benchmarks.utils import make_classification_dataset


def get_cladc_train(root: str, transform: Callable = None, img_size: int = 64, avalanche: bool = False, sequence_type: str = "source",
                    end_with_source_domain: bool = False) -> Sequence[CladClassification]:
    """
    Returns a sequence of training sets that are chronologically ordered, defined as in the ICCV '21 challenge.

    :param root: root path to the dataset
    :param transform: a callable transformation for the data images
    :param img_size: the width/height of the images, default is 64 by 64.
    :param avalanche: If true, this will return AvalancheDataset objects.
    :param sequence_type: 'source', 'tta' or 'all'
    """
    task_dicts = [{'date': '20191111', 'period': 'Daytime'},
                  {'date': '20191111', 'period': 'Night'},
                  {'date': '20191117', 'period': 'Daytime'},
                  {'date': '20191117', 'period': 'Night'},
                  {'date': '20191120', 'period': 'Daytime'},
                  {'date': '20191121', 'period': 'Night'}, ]
    
    if sequence_type == "source":
        task_dicts = [task_dicts[0]]
    elif sequence_type == "tta":
        task_dicts = task_dicts[1:]
        if end_with_source_domain:
            task_dicts.append(task_dicts[0])
    elif sequence_type == "all":
        pass
    else:
        raise ValueError(f"Unknown sequence type: {sequence_type}")

    match_fn = (create_match_dict_fn(td) for td in task_dicts)

    # All training images are part of the original validation set of SODA10M.
    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_val.json')
    train_sets = [get_matching_classification_set(root, annot_file, mf, img_size=img_size, transform=transform) for mf
                  in match_fn]

    for ts in train_sets:
        ts.chronological_sort()

    if avalanche:
        avalanche_datasets = []
        for i, train_set in enumerate(train_sets):
            if i == len(train_sets) - 1:
                # source domain label
                task_label = 0
            else:
                task_label = i + 1

            avalanche_datasets.append(make_classification_dataset(train_set, task_labels=task_label))
        return avalanche_datasets
    else:
        return train_sets


def get_cladc_val(root: str, transform: Callable = None, img_size: int = 64, avalanche=False) -> CladClassification:
    """
    Returns the default validation set of the ICCV '21 benchmark.
    """

    def val_match_fn_1(obj_id, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj_id]['image_id']]
        date = img_annot["date"]
        return not (date == "20191120" or date == "20191117" or date == "20191111" or
                    (date == "20191121" and img_annot['period'] == "Night"))

    def val_match_fn_2(obj_id, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj_id]['image_id']]
        time = img_annot['time']
        date = img_annot["date"]
        label = obj_dic[obj_id]['category_id']
        return label or (label and date == "20181015" and (time == '152030' or time == '160000'))

    annot_file_1 = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_val.json')
    annot_file_2 = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_train.json')

    val_set = ConcatDataset([
        get_matching_classification_set(root, annot_file_1, val_match_fn_1, img_size=img_size, transform=transform),
        get_matching_classification_set(root, annot_file_2, val_match_fn_2, img_size=img_size, transform=transform)])

    if avalanche:
        val_set.targets = val_set.datasets[0].targets + val_set.datasets[1].targets
        return [make_classification_dataset(val_set, task_labels=0)]
    else:
        return val_set


def get_cladc_test(root: str, transform=None, img_size: int = 64, avalanche=False):
    """
    Returns the full test set of the CLAD-C benchmark. Some domains are overly represented, so for a fair
    evaluation see get_cladc_domain_test.
    """

    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_test.json')
    test_set = get_matching_classification_set(root, annot_file, lambda *args: True, img_size=img_size,
                                               transform=transform)

    if avalanche:
        return [make_classification_dataset(test_set, task_labels=0)]
    else:
        return test_set


def get_cladc_domain_test(root: str, transform: Callable = None, img_size: int = 64, avalanche=False):
    """
    Returns fine-grained domain sets of the test set for each available combination, note that not all the
    combinations of domains exist.
    """
    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_test.json')
    test_sets = get_cladc_domain_sets(root, annot_file, ["period", "weather", "city", "location"],
                                      img_size=img_size, transform=transform)

    if avalanche:
        from avalanche.benchmarks.utils import AvalancheDataset
        # return [AvalancheDataset(ts) for ts in test_sets if len(ts) > 0]
        return [ts for ts in test_sets if len(ts) > 0]
    else:
        return [ts for ts in test_sets if len(ts) > 0]


def cladc_avalanche(root: str, train_trasform: Callable = None, test_transform: Callable = None, img_size: int = 64,
                    end_with_source_domain: bool = False):
    """
    Creates an Avalanche benchmark for CLADC, with the default Avalanche functinalities.
    """
    from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark

    train_sets = get_cladc_train(root, train_trasform, img_size, avalanche=True, sequence_type="tta", end_with_source_domain=end_with_source_domain)
    test_sets = get_cladc_test(root, test_transform, img_size, avalanche=True)
    val_sets = get_cladc_val(root, test_transform, img_size, avalanche=True)

    return create_multi_dataset_generic_benchmark(train_datasets=train_sets, test_datasets=test_sets,
                                                  other_streams_datasets={"val": val_sets})
