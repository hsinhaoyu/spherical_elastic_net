import tensorflow as tf
import numpy as np

map_w = 8               # the width of the net
map_h = 8               # the height of the net
n_prototypes = 200      # number of prototypes on the sphere

b = 0.0088              # weighting of the regulaization term
eta = 0.1               # initial learning rate

k0 = 0.6                # initial value for kappa
k_decay = 0.00205       # decay rate for kappa

def polar2cartesian(theta, phi):
    return (
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi)
    )

def generate_prototypes(n):
    """
        ref: "How to generate equidistributed points on the surface of a sphere" by Markus Deserno
        note that in the paper, {phi, theta} correspond to {theta, phi} in the rest of this code
        n is an approximate number. the number of returned points are smaller
    """
    a = 2.0*3.1415926/n
    d = np.sqrt(a)
    mtheta = int(round(0.5*3.1415926/d))
    dtheta = 0.5*3.1415926/mtheta
    dphi = a/dtheta

    lst = []
    for m in range(mtheta - 1):
        theta = 0.5*3.1415926 * (m + 0.5)/mtheta
        mphi = int(round(2.0*3.1415926*np.sin(theta)/dphi))
        for n in range(mphi):
            phi = 2.0*3.1415926*n/mphi
            lst.append(polar2cartesian(phi, theta))
    return lst

def tensor_polar2cartesian(x):
    """
        x: tensor of the shape [size, 2] (polar coordinate)
        return: tensor of the shape [size, 3] (cartesian coordinate)
    """
    return tf.stack(
        [tf.cos(x[:, 0]) * tf.sin(x[:, 1]),
         tf.sin(x[:, 0]) * tf.sin(x[:, 1]),
         tf.cos(x[:, 1])],
        axis=-1
    )

def neighborhood(m, n):
    """
        numbers {0...m*n-1} arranged in a mxn grid, produce a list of neighboring numbers.
        The ith element of the list is a list of numbers that neighbor i
    """
    x = np.array(range(m*n)).reshape([m, n])
    lst = []
    for i in range(m):
        for j in range(n):
            neighbors = []
            for ii in [-1, 0, 1]:
                for jj in [-1, 0, 1]:
                    if (i+ii>=0) and (i+ii<m) and (j+jj>=0) and (j+jj<n):
                        neighbors.append(x[i+ii,j+jj])
            lst.append(neighbors)
    return lst

def make_mask(m, n, neigbors):
    x = np.zeros([m*n, m*n], dtype=np.float64)
    for i in range(m*n):
        for j in neigbors[i]:
            x[i, j] = 1.0
    return x

def initial_condition(map_w, map_h):
    """
        uniform random sample map_w * map_h points on the sphere
    """
    map_size = map_w * map_h
    lst = []

    for i in range(map_size):
        u = np.random.random_sample()
        v = np.random.random_sample()

        lst.append([
            2.0 * 3.1415926 * u,
            np.arccos(v)
        ])
    return np.array(lst, dtype=np.float64)

#########################
# set up the variables
#########################

# concentration parameter for the Kent distribution
# smaller kappa -> more concentrated distribition
# meaningful range of kappa: 1.0 to 0.001
kappa = tf.placeholder(tf.float64, shape=(), name="kappa")

# the weight of the regularization term
beta = tf.placeholder(tf.float64, shape=(), name="beta")

#### prototypes. these are in cartesian coordinates
x  = tf.constant(
        generate_prototypes(n_prototypes),
        dtype=tf.float64)

#### train these points on the sphere.
y_polar = tf.get_variable(
            "y_polar",
            dtype = tf.float64,
            initializer = initial_condition(map_w, map_h))

y = tensor_polar2cartesian(y_polar)

#### regularization term

# n is a list of map_w * map_h objects. The i-th item of n is a list containing the indices of nodes neighboring the i-th node
n = neighborhood(map_w, map_h)

mask = make_mask(map_w, map_h, n)

# pairwise dot product
yy_dot = tf.reduce_sum(
        tf.multiply(
            tf.expand_dims(y, 1),
            tf.expand_dims(y, 0)),
        axis=-1)

# clip the dot products because the acos() needed to calculate geodesic distance
# is unstale at -1.0 and 1.0 (ie. gradient becomes -Infinity)
yy_dot = tf.clip_by_value(yy_dot, -0.99999, 0.99999)

yy_distance = tf.acos(yy_dot)

# mask entries for non-neighboring points on the net
yy_distance_masked = tf.multiply(mask, yy_distance)

regularizer = tf.reduce_sum(yy_distance_masked)

#### the energy function
# calculate the dot products of all pairs of point between prototypes and x
# by brocasting multiply
xy_dot = tf.reduce_sum(
        tf.multiply(
            tf.expand_dims(x, 1),
            tf.expand_dims(y, 0)),
        axis=-1)

# apply the kent distribution
# smaller kappa -> more concentrated distribition
pairwise_kent = tf.exp(1.0/kappa * xy_dot)

# first sum over y, take log, then sum again
term1 = tf.reduce_sum(
            tf.log(
                tf.reduce_sum(pairwise_kent, axis=1)))

energy_term1 = -1.0 * kappa * term1
energy_term2 = beta * regularizer

# note that this term is only use for debugging. It's not used to calculate gradient
energy = energy_term1 + energy_term2

tf.summary.scalar("term1", energy_term1)
tf.summary.scalar("term2", energy_term2)
tf.summary.scalar("energy", energy)

#### gradient operations
# basis for the tangent space
# length of vector is 0.1
eTheta = tf.stack([
            -0.1 * tf.sin(y_polar[:, 0]),
            0.1 * tf.cos(y_polar[:, 0]),
            0.1 * tf.zeros((map_w * map_h,), dtype=tf.float64)],
         axis=-1)

ePhi = tf.stack([
            0.1 * tf.cos(y_polar[:, 0]) * tf.cos(y_polar[:, 1]),
            0.1 * tf.sin(y_polar[:, 0]) * tf.cos(y_polar[:, 1]),
            -0.1 * tf.sin(y_polar[:, 1])],
        axis=-1)

global_step = tf.Variable(0, trainable=False)
update_global_step = tf.assign_add(global_step, 1)

opt = tf.train.GradientDescentOptimizer(learning_rate = eta)

polar_grad_1 = opt.compute_gradients(energy_term1, var_list = [y_polar])[0][0]
polar_grad_2 = opt.compute_gradients(energy_term2, var_list = [y_polar])[0][0]

polar_grad = polar_grad_1 + polar_grad_2

# calculate the gradient in 3D space
# take the gradients caclulated by Tensorflow, apply the basis {eTheta, ePhi}
delta_y = tf.multiply(
                tf.expand_dims(polar_grad[:,0], axis=-1),
                eTheta) + \
             tf.multiply(
                tf.expand_dims(polar_grad[:,1], axis=-1),
                ePhi)

# updated y in cartesian space
y_updated = y - delta_y

# translate to spherical coordinates
y_polar_updated_theta = tf.atan2(y_updated[:,1], y_updated[:,0])

y_polar_updated_phi0 = y_updated[:,2]/tf.sqrt(y_updated[:,0]*y_updated[:,0] + y_updated[:,1]*y_updated[:,1] + y_updated[:,2]*y_updated[:, 2])

# due to numerical error, y_polar_updated_phi0 can go slighly larger than 1.0, which causes acos() to return nan
# set them to 1.0
y_polar_updated_phi = tf.where(
                        tf.greater(y_polar_updated_phi0, 1.0),
                        tf.zeros(y_polar_updated_phi0.get_shape(), dtype=tf.float64)+1.0,
                        y_polar_updated_phi0)

y_polar_updated = tf.stack([
    y_polar_updated_theta,
    tf.acos(y_polar_updated_phi)],
    axis=-1)

update_y_polar = tf.assign(y_polar, y_polar_updated)

####################
# let's do it
####################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(dir_name, s.graph)

k = k0

np.savetxt("prototypes.data", sess.run(x))

fp = open("neighbohood.data", "w")
for i in n:
    for j in i:
        fp.write("%d "%j)
    fp.write("\n")
fp.close()


for i in range(5000):

    print i, ":", k
    if i<100:
        zz = sess.run(y_polar)
        np.savetxt("saves/y-"+str(i).zfill(5)+".data", zz)
    if (i>100) and (i%10==0):
        zz = sess.run(y_polar)
        np.savetxt("saves/y-"+str(i).zfill(5)+".data", zz)

    [e1, e2, e3] = sess.run([energy, energy_term1, energy_term2], {kappa: k, beta: b})
    print "    %8.7f %8.7f %8.7f"%(e1, e2, e3)
    sess.run(update_y_polar, {kappa: k, beta: b})
    sess.run(update_global_step)
    if k>0.002:
        k = k - k * k_decay
