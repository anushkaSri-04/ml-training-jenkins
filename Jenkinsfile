pipeline {
    agent any

    environment {
        PYTHON_ENV = 'ml-training-env'
    }

    parameters {
        choice(name: 'ML_FRAMEWORK',
               choices: ['sklearn'],
               description: 'Select ML framework for training')
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                sh '''
                    python3 -m venv ${PYTHON_ENV}
                    . ${PYTHON_ENV}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Data Validation') {
            steps {
                sh '''
                    . ${PYTHON_ENV}/bin/activate
                    python3 -c "import pandas as pd; df = pd.read_csv('data/clean_dataset.csv'); print('Data shape:', df.shape)"
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    . ${PYTHON_ENV}/bin/activate
                    python3 src/train_sklearn.py
                '''
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'models/**/*', fingerprint: true
            }
        }
    }

    post {
        success {
            echo 'Training completed successfully!'
        }
        failure {
            echo 'Training failed!'
        }
    }
}