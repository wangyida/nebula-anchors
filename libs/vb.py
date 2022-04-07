import tensorflow as tf
from libs import utils


def variational_bayes(h, n_code):
    """Summary

    Parameters
    ----------
    h : TYPE
        Description
    n_code : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    z_mu = utils.linear(h, n_code, name='mu')[0]
    z_log_sigma = 0.5 * utils.linear(h, n_code, name='log_sigma')[0]

    # Sample from noise distribution p(eps) ~ N(0, 1)
    epsilon = tf.random_normal(tf.stack([tf.shape(h)[0], n_code]))

    # Sample from posterior
    z = tf.add(z_mu, tf.multiply(epsilon, tf.exp(z_log_sigma)), name='z')
    # -log(p(z)/q(z|x)), bits by coding.
    # variational bound coding costs kl(p(z|x)||q(z|x))
    # d_kl(q(z|x)||p(z))
    loss_z = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)
    return z, z_mu, z_log_sigma, loss_z
