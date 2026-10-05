"""Run the full pipeline: train -> predict -> validate outputs."""
import sys

import config
import predict
import train_model
import validate_outputs


def main():
    print("=" * 60)
    print("AMAZON ML CHALLENGE - BUSINESS ENTITY RESOLUTION")
    print(f"FIRST {config.LIMIT:,} RECORDS")
    print("=" * 60)
    train_model.main()
    predict.main()
    if not validate_outputs.validate():
        sys.exit(1)


if __name__ == "__main__":
    main()
