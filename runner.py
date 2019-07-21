import numpy as np
import scipy.io as sio

from test_utils import plot_sgd_results
from neural_network import rnn_model

if __name__ == "__main__":
    data = sio.loadmat('resources/PeaksData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size = trainX.shape[0]

    layers_count = 5
    batch_size = 100
    learning_rate = 0.03
    iterations = 200
    freq = 2

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)
    theta = np.array([0.827031962161076, -0.614886009271391, -1.23514984542835, 1.29132482759806, 0.0756490955935271,
                      0.117466420711722, -1.92789550462287, 0.993562919712172, -1.48035059106780, 0.466980593109596,
                      -0.657707841221089, -0.512293719155152, -0.582736043994383, 1.08282811067222, -1.29156050026331,
                      -0.378277444320590, 1.03607002343218, 1.38547245702994, 1.09145827007654, -0.163790382207854,
                      0.441309707284371, -0.746316127923388, -1.40371248632631, -0.0721897475362090, 0.229715657373079,
                      0.938338911352737, -0.235338535051665, 1.08190665003805, -0.426220450678048, 2.01732105458847,
                      0.239270574567162, 0.459980638568151, 0.903047019477418, -0.283065805309313, 0.905037384131715,
                      0.0759292255178730, -0.695565398072913, -0.756020430309528, 0.280239354848502, -1.06203003404339,
                      0.381516044872692, 1.75201148156847, -0.260967827473427, -0.913493059660084, 0.616331768879268,
                      -1.31606176586980, -0.855453162645312, -0.147051677854348, -1.35536891275582, 0.513187983387222,
                      0.719485324475787, -1.05228510613148, -1.43680372535555, 0.925452730798893, 0.637063739856243,
                      -0.717119885050594, -0.351550407764129, -1.54219566565802, 0.0556388411419630, 0.441792643057164,
                      0.311109786255297, -0.585016868726695, 1.29661261097353, -1.15352023707661,
                      1.55917743323409]).reshape(65, 1)

    model = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)
    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    plot_sgd_results(iterations, train_loss, train_accuracy, test_loss, test_accuracy, 'SGD results (PeaksData)', 'q5_peaks_data', True)
