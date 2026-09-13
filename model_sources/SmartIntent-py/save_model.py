import argparse
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(
        description='Load and save a TensorFlow model.')
    parser.add_argument('--model_path', required=True,
                        help='Path to the model file (H5 format)')
    parser.add_argument('--model_dir', required=True,
                        help='Directory to save the TensorFlow model')

    args = parser.parse_args()

    # 加载H5模型
    model = tf.keras.models.load_model(args.model_path)

    # 保存模型
    tf.saved_model.save(model, args.model_dir)


if __name__ == "__main__":
    main()
