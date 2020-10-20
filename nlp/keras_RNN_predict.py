import tensorflow as tf
from keras_RNN import embeddings, letters, build_model

def main():
    model = load_model()

    while True:
        starting_string = input()
        if starting_string == 'quit':
            break
        if starting_string == '':
            print("Enter seed text")
            continue

        print("Generating...")
        text = generate_text(model, starting_string)
        print(text)

def load_model():
    model = build_model(batch_size=1)
    model.load_weights(tf.train.latest_checkpoint('./checkpoints'))
    model.build(tf.TensorShape([1, None]))

    model.summary()
    return model


def generate_text(model, start_string):
    num_generate = 2000

    input_eval = [embeddings[letter] for letter in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    model.reset_states()
    text_generated = []

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predicted_id = tf.random.categorical(
                predictions,
                num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(letters[predicted_id])

    return start_string + "".join(text_generated)

if __name__ == '__main__':
    main()
