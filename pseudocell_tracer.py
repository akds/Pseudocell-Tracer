import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import spearmanr, pearsonr

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams["figure.figsize"] = (8,8)

import seaborn as sns
from umap import UMAP
import configparser as cp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###################################################
# Read in configuration file for network parameters
###################################################

config = cp.ConfigParser()
config.read('config.py')

rna_hidden_dim_1 = int(config.get('AE', 'RNA_Layer1'))
rna_hidden_dim_2 = int(config.get('AE', 'RNA_Layer2'))
rna_latent_dim = int(config.get('AE', 'RNA_Latent'))
translate_hidden_dim = int(config.get('AE', 'Translate_Layer1'))
l2_norm_dec = float(config.get('AE', 'L2_Norm_AE'))
l2_norm_kl = float(config.get('AE', 'L2_Norm_KL'))
learning_rate_kl = float(config.get('AE', 'Learning_Rate_KL'))
learning_rate_ae = float(config.get('AE', 'Learning_Rate_AE'))

Z_dim = int(config.get('CGAN', 'Z_dim'))
gen_dim1 = int(config.get('CGAN', 'Gen_Layer1'))
gen_dim2 = int(config.get('CGAN', 'Gen_Layer2'))
det_dim = int(config.get('CGAN', 'Det_Layer1'))
learning_rate_cgan = float(config.get('CGAN', 'Learning_Rate'))
l2_norm_cgan = float(config.get('CGAN', 'L2_Lambda'))
epoch_cgan = int(config.get('CGAN', 'Max_Epoch'))

###################################################
# Read in command line arguments
###################################################

parser = argparse.ArgumentParser(description='Perform Pseudocell Tracer Algorithm')

parser.add_argument('-d', '--data', help='Tab delimited file representing matrix of samples by genes', required=True)
parser.add_argument('-s', '--side_data', help= 'Tab delimited file for side information to be used', required=True)
parser.add_argument('-o', '--output', help='Output directory', required=True)

parser.add_argument('-p', '--plot_style', help= 'Plotting style to be used (UMAP or tSNE)', default="UMAP")
parser.add_argument('-n', '--num_cells_gen', help='Number of pseudocells to generate at each step', default=100, type=int)
parser.add_argument('-k', '--num_steps', help='Number of pseudocell states', default=100, type=int)

parser.add_argument('-a', '--start_states', help='Number of pseudocell states', nargs="+", required=True)
parser.add_argument('-b', '--end_states', help='Number of pseudocell states', nargs="+", required=True)
parser.add_argument('-g', '--genes_to_plot', help='Genes to plot in pseudocell trajectory', nargs="+")

args = parser.parse_args()

dset = args.data
sset = args.side_data
out = args.output
plot_method = args.plot_style
num_cells_gen = args.num_cells_gen
num_steps = args.num_steps
start_states = args.start_states
end_states = args.end_states
genes_to_plot = args.genes_to_plot

###################################################
# Function to plot
###################################################

def plot_data(rna, side):

    if plot_method == "tSNE":
        tsne = TSNE()
        results = np.array(tsne.fit_transform(rna.values))
        
    if plot_method == "UMAP":
        umap = UMAP()
        results = np.array(umap.fit_transform(rna.values))

    plot_df = pd.DataFrame(data=results, columns=["X","Y"])
    cl = np.argmax(side.values, axis=1)
    plot_df["subtype"] = [side.columns[x] for x in cl]
    
    return sns.scatterplot(x="X", y="Y", hue="subtype", data=plot_df)
    
    
###################################################
# Load Data
###################################################

rna_data = pd.read_csv(dset, header=0, index_col=0, sep="\t")
side_data = pd.read_csv(sset, index_col=0, header=0, sep="\t")


###################################################
# Filter common samples
###################################################

sample_list = np.intersect1d(rna_data.columns.values, side_data.columns.values)
gene_list = rna_data.index.values

side_data = side_data.filter(sample_list, axis=1)
rna_data = rna_data.filter(sample_list, axis=1)

rna_data = rna_data.transpose()
side_data = side_data.transpose()

print("Loaded Data...")


###################################################
# Create output directory
###################################################
print("Creating output directory...")

try: 
    os.mkdir("results")
except:
    pass
    
out_dir = "results/" + out

try:
    os.mkdir(out_dir)
    print("Directory " , out_dir ,  " Created...") 
except FileExistsError:
    print("Warning: Directory " , out ,  " already exists...")

with open(out_dir + '/run_parameters.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    
with open(out_dir + '/run_network_config.txt', 'w') as f:
    config.write(f)

    
###################################################
# Plot input data
###################################################

scatter = plot_data(rna_data, side_data)
plt.title("Input Data (" + str(plot_method) + ")")
plt.savefig(out_dir + "/input_scatter.png")
plt.clf()


###################################################
# Train supervised encoder
###################################################

print("Training supervised encoder...")

scaler = StandardScaler().fit(rna_data.values)
norm_rna_data = np.clip(scaler.transform(rna_data.values),-3,3)
  
reg_kl = tf.keras.regularizers.l2(l2_norm_kl)
kl_model = tf.keras.Sequential([

    tf.keras.layers.Dense(rna_hidden_dim_1, activation='relu',
        kernel_regularizer=reg_kl, input_shape=(len(gene_list),)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(rna_hidden_dim_2, activation='relu',
        kernel_regularizer=reg_kl),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(rna_latent_dim, activation='sigmoid', name='latent_layer',
        kernel_regularizer=reg_kl),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(translate_hidden_dim, activation='relu',
        kernel_regularizer=reg_kl),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(side_data.values.shape[1], activation=tf.nn.softmax,
        kernel_regularizer=reg_kl, name='relative_prediction')
])

es_cb_kl = tf.keras.callbacks.EarlyStopping('val_kullback_leibler_divergence', patience=100, restore_best_weights=True)
kl_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate_kl), loss=tf.keras.losses.KLD, metrics=['kullback_leibler_divergence'])
kl_model.fit(norm_rna_data, side_data.values, epochs=10000, callbacks=[es_cb_kl], validation_split=0.1, verbose=0, batch_size=1024)
kl_model.fit(norm_rna_data, side_data.values, epochs=5, verbose=0, batch_size=1024)
kl_model.save(out_dir + "/encoder.h5")

###################################################
# Get latent data
###################################################

print("Getting latent data...")

latent_model = tf.keras.Model(inputs=kl_model.input, outputs=kl_model.get_layer('latent_layer').output)
latent = latent_model.predict(norm_rna_data)
latent_df = pd.DataFrame(data=latent, index=sample_list)
latent_df.to_csv("latent_values.tsv", sep="\t")

scatter = plot_data(latent_df, side_data)
plt.title("Latent Data (" + str(plot_method) + ")")
plt.savefig(out_dir + "/latent_scatter.png")
plt.clf()


###################################################
# Train decoder
###################################################

print("Training decoder...")

reg_dec = tf.keras.regularizers.l2(l2_norm_dec)

dec_model = tf.keras.Sequential([
    tf.keras.layers.Dense(rna_hidden_dim_2, activation='relu', input_shape=(rna_latent_dim,),
        kernel_regularizer=reg_dec, bias_regularizer=reg_dec),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(rna_hidden_dim_1, activation='relu',
        kernel_regularizer=reg_dec, bias_regularizer=reg_dec),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(len(gene_list), activation=None, name='rna_reconstruction',
        kernel_regularizer=reg_dec, bias_regularizer=reg_dec)
])

es_cb_dec = tf.keras.callbacks.EarlyStopping('val_mean_squared_error', patience=100, restore_best_weights=True)
dec_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate_ae), loss=tf.keras.losses.MSE, metrics=['mse'])
dec_model.fit(latent, norm_rna_data, epochs=100000, callbacks=[es_cb_dec], validation_split=0.1, 
              verbose=0, batch_size=1024)
dec_model.fit(latent, norm_rna_data, epochs=5, verbose=0, batch_size=1024)
dec_model.save(out_dir + "/decoder.h5")


###################################################
# Get reconstructed values
###################################################

decoded = dec_model.predict(latent)
decoded_df = pd.DataFrame(data=decoded, index=sample_list, columns=gene_list)
decoded_df.to_csv("reconstructed_values.tsv", sep="\t")

scatter = plot_data(decoded_df, side_data)
plt.title("Reconstructed Data (" + str(plot_method) + ")")
plt.savefig(out_dir + "/reconstructed_scatter.png")
plt.clf()

###################################################
# Build CGAN Architecture
###################################################

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, rna_latent_dim])
y = tf.placeholder(tf.float32, shape=[None, side_data.shape[1]])

D_W1 = tf.Variable(xavier_init([rna_latent_dim + side_data.shape[1], det_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[det_dim]))

D_W2 = tf.Variable(xavier_init([det_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

""" Generator Net model """

Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + side_data.shape[1], gen_dim1]))
G_b1 = tf.Variable(tf.zeros(shape=[gen_dim1]))

G_W2 = tf.Variable(xavier_init([gen_dim1, gen_dim2]))
G_b2 = tf.Variable(tf.zeros(shape=[gen_dim2]))

G_W3 = tf.Variable(xavier_init([gen_dim2, rna_latent_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[rna_latent_dim]))

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_log_prob, G_prob

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])



###################################################
# Build CGAN Architecture
###################################################

print("Training CGAN...")

reg_dec = tf.keras.regularizers.l2(l2_norm_cgan)

G_sample, G_sample_sigmoid = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate_cgan).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate_cgan).minimize(G_loss, var_list=theta_G)

train_side = side_data.values
train_latent = latent

train = tf.data.Dataset.from_tensor_slices((train_latent, train_side))
train = train.batch(1024)
train = train.shuffle(1000000)

train_iterator = train.make_initializable_iterator()
train_next_element = train_iterator.get_next()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for it in range(epoch_cgan):

    sess.run(train_iterator.initializer)
    batches = 0
    total_g_loss = 0
    total_d_loss = 0
    while True:
        try:
            X_mb, y_mb = sess.run(train_next_element)
            noise = np.random.normal(loc=0,scale=0.02,size=y_mb.shape[0]*y_mb.shape[1])
            y_mb = np.clip(y_mb + noise.reshape(y_mb.shape[0], y_mb.shape[1]),0,1)
            sums = np.sum(y_mb,1)
            y_mb = y_mb/sums.reshape(-1,1)
            batches += 1
            Z_sample = sample_Z(X_mb.shape[0], Z_dim)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
            _, G_loss_curr, samp = sess.run([G_solver, G_loss, G_sample], feed_dict={Z: Z_sample, y:y_mb})
        except tf.errors.OutOfRangeError:
            print(" %05d\tGen Loss: %.3f\tDisc Loss: %.3f" % (it, G_loss_curr, D_loss_curr), end="\r")
            break


z_sample = sample_Z(side_data.values.shape[0], Z_dim)
gen_data = sess.run(G_sample, feed_dict={Z: z_sample, y:side_data.values})            
gen_latent_df = pd.DataFrame(data=gen_data, index=sample_list)
gen_latent_df.to_csv("observed_generated_latent_values.tsv", sep="\t")
 
scatter = plot_data(gen_latent_df, side_data)
plt.title("Generated Latent Data (" + str(plot_method) + ")")
plt.savefig(out_dir + "/generated_latent_scatter.png")
plt.clf()

gen_decoded = dec_model.predict(gen_data)
gen_decoded_df = pd.DataFrame(data=gen_decoded, index=sample_list, columns=gene_list)

scatter = plot_data(gen_decoded_df, side_data)
plt.title("Generated Decoded Data (" + str(plot_method) + ")")
plt.savefig(out_dir + "/generated_decoded_scatter.png")
plt.clf()

###################################################
# Generate pseudocell trajectories
###################################################

print("\nGenerating data...")

states = side_data.columns.values

for c1_state in start_states:
    for c2_state in end_states:  
        y_sample = []

        for i in range(0,num_steps + 1):
            for j in range(num_cells_gen):
                c1 = 1 - i/num_steps
                c2 = i/num_steps
                c1_idx = np.where(states==c1_state)
                c2_idx = np.where(states==c2_state)
                y_state = np.zeros(len(states))
                y_state[c1_idx] = c1
                y_state[c2_idx] = c2
                y_sample.append(y_state)
            
        y_sample = np.array(y_sample)
        z_sample = sample_Z(y_sample.shape[0], Z_dim)
        gen_data = sess.run(G_sample, feed_dict={Z: z_sample, y:y_sample})
        data_df = pd.DataFrame(gen_data, index=range((num_steps + 1) * num_cells_gen))
        y_df = pd.DataFrame(y_sample, index=range((num_steps + 1) * num_cells_gen), columns=states)

        gen_dir = out_dir + "/" + c1_state + "_" + c2_state
        
        try: 
            os.mkdir(gen_dir)
        except:
            pass
        
        data_df.to_csv(gen_dir + "/generated_latent_data.tsv", sep="\t")
        y_df.to_csv(gen_dir + "/generated_side_data.tsv", sep="\t")
        
        dec_df = pd.DataFrame(dec_model.predict(gen_data), index=range((num_steps + 1) * num_cells_gen), columns=rna_data.columns)
        dec_df.to_csv(gen_dir + "/generated_data.tsv", sep="\t")

                                    
        g_idx = 0
        full_genes_to_plot = np.concatenate((genes_to_plot, c1_state, c2_state), axis=None)
        
        gene_plot_df = pd.DataFrame(data=np.zeros(((num_steps + 1) * num_cells_gen * len(full_genes_to_plot), 3)), 
                                    index=range((num_steps + 1) * num_cells_gen * len(full_genes_to_plot)), 
                                    columns=["State", "Gene", "Value"])
        
        for g in full_genes_to_plot:
            if g in gene_list:
                offset = (num_steps + 1) * num_cells_gen * g_idx
                for k in range((num_steps + 1) * num_cells_gen):
                    state_percent = y_df.iloc[k][c2_state]
                    gene_value = dec_df.iloc[k][g]
                    gene_plot_df.iloc[offset + k] = [state_percent, g, gene_value]
                g_idx += 1
            sns.lineplot(x="State", y="Value", hue="Gene", data=gene_plot_df)
            plt.title(c1_state + " to " + c2_state)
            plt.xlabel("% " + c2_state)
            plt.savefig(gen_dir + "/genes.png")
            plt.clf()