import mlflow
import configparser
from src.models.run_experiment_from_config import Experiment
from pathlib import Path
import numpy as np

def run_stability_test(server_uri, top_n_models, experiment_name, by_parameter, file, epochs):
    runs_params = []
    parameters = ['params.Batch size', 'params.recalculate_period', 'params.number of dimensions',
                  'params.learning_rate', 'metrics.Training time']
    mlflow.set_tracking_uri(server_uri)
    exp = mlflow.search_runs(experiment_names=[experiment_name])
    batch_sizes = exp[by_parameter].unique()
    exp.dropna(subset=['metrics.metrics.MultiClassAccuracy'])

    for b in np.flip(batch_sizes):
        tmp_exp = exp[exp[by_parameter] == b]
        tmp_exp = tmp_exp.sort_values(by=['metrics.metrics.MultiClassAccuracy'],  ascending=False)
        tmp_exp = tmp_exp.head(n=top_n_models)
        for index, row in tmp_exp.iterrows():
            params = {}
            for p in parameters:
                params[p] = row[p]
            runs_params.append(params)
    for i in runs_params:
        set_value_in_property_file(file, i['params.Batch size'], i['params.recalculate_period'],
                                   i['params.number of dimensions'], i['params.learning_rate'], epochs)
        experiment = Experiment(file)
        experiment.build_objects()
        experiment.build_metrics()
        experiment.train_and_log_model()
        del experiment



def set_value_in_property_file(file_path, batch_size, recalculate_period, out_dim, learning_rate, epochs):

    config = configparser.ConfigParser()
    config.read(file_path)
    config.set('training', 'batch_size', str(batch_size))
    config.set('training', 'recalculate_period', str(recalculate_period))
    config.set('training', 'lr', str(learning_rate))
    config.set('training', 'out_dim', str(out_dim))
    config.set('training', 'epochs', str(epochs))
    cfgfile = open(file_path,'w')
    config.write(cfgfile, space_around_delimiters=False)
    cfgfile.close()


if __name__ == "__main__":
    run_stability_test("http://127.0.0.1:5000", 2, 'grid_search_one_epoch', 'params.Batch size',
                       Path("../configs/visual_config_davis_bouldin_centroid_freeze.ini"),
                       3)