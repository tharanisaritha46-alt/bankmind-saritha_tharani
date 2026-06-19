"""Quick API smoke test for reviewers.

Run after installing requirements:
    python smoke_test.py
"""
from api.app import Customer, explain, health, predict


def main() -> None:
    print("GET /health ->", health())

    customer = Customer(
        age=64,
        job="retired",
        balance=7000,
        housing="no",
        loan="no",
        duration=600,
    )

    print("POST /predict ->", predict(customer).model_dump())
    print("POST /explain ->", explain(customer).model_dump())


if __name__ == "__main__":
    main()
