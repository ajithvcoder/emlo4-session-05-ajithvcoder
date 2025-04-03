pipeline {
  agent any 
  environment {
      CONFIG_FILE = ''
  }
  stages {
    stage('Load Config') {
        steps {
            script {
                def config = load CONFIG_FILE
                echo "Loaded configuration for ${params.ENV} environment"
            }
        }
    }
    stage("build") {
      steps {
        echo "building application"
      }
    }
    stage('Load Config') {
        steps {
            script {
                def config = load CONFIG_FILE
                echo "Loaded configuration for ${params.ENV} environment"
            }
        }
    }

    stage('Deploy') {
        steps {
            script {
                def config = load CONFIG_FILE
                // Use the loaded configuration for deployment
                echo "Deploying to ${params.ENV} environment with URL: ${config.url}"
                // Add your deployment steps here
            }
        }
    }
    stage("test_old") {
      steps {
        echo "testing application"
      }
    }
    stage("deploy_old") {
      steps {
        echo "deploying application"
      }
    }
  }

  parameters {
      choice(name: 'ENV', choices: ['dev', 'prod'], description: 'Select the environment to deploy', defaultValue: 'dev')
  }
}
