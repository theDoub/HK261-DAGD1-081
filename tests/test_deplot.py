def main():
    from src.models.deplot import DePlotHarness

    model = DePlotHarness()

    image_path = "data/raw/chart-for-test.jpg"

    result = model.predict(image_path)

    print("DePlot output:")
    print(result)


if __name__ == "__main__":
    main()