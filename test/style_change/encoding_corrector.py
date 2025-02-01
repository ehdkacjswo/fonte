from charset_normalizer import from_path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str)
    args = parser.parse_args()

    encoded_code = from_path(args.path).best()

    if encoded_code is not None:
        with open(args.path, 'w', encoding='utf-8') as file:
            file.write(str(encoded_code))