import tensorflow as tf


def binary_crossentropy_loss(y_true, y_pred):
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce_loss(y_true, y_pred)


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def content_loss(y_true, y_pred):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    vgg.trainable = False
    vgg_pred_output = vgg(y_pred)
    vgg_true_output = vgg(y_true)
    return tf.reduce_mean(tf.square(vgg_pred_output - vgg_true_output))


def generator_loss(y_true, y_pred, discriminator_output):
    valid_labels = tf.ones_like(discriminator_output)  # Target real label for generator
    adversarial_loss = binary_crossentropy_loss(valid_labels, discriminator_output)

    c_loss = 5 * content_loss(y_true, y_pred)  # Content loss using VGG
    r_loss = 10 * mse_loss(y_true, y_pred)
    return adversarial_loss, c_loss, r_loss


def discriminator_loss(real_labels, fake_labels, real_output, fake_output):
    real_loss = binary_crossentropy_loss(real_labels, real_output)
    fake_loss = binary_crossentropy_loss(fake_labels, fake_output)
    return (real_loss + fake_loss) / 2