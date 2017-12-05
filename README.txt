HOW TO RUN
==========

requirements:
    - python3
    - virtualenv

Step 1: Create environment

    virtualenv env -p python3

Step 2: Install requitements

    pip install -r requirements.txt

Step3: Run

    source env/bin/activate
    # Run analysis for subtask A
    python svm.py 1

    # Run analysis for subtask B
    python svm.py 2

    # Run analysis for subtask C
    python ordinalRegression.py



RUNNING WITH DIFFRENT DATA
==========================

To run with different data, replace the files in semEval_train_2016 with the new data.
Make sure to keep the same names since otherwise an error will occure.
