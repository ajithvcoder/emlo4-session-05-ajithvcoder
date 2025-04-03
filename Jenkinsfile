pipeline {
  agent any 
  stages {

    stage('Select Environment') {
        steps {
            script {
                if (params.ENV == 'prod') {
                    CONFIG_FILE = 'prod.groovy'
                } else {
                    CONFIG_FILE = 'dev.groovy'
                }
            }
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
      choice(name: 'ENV', choices: ['dev', 'prod'], description: 'Select the environment to deploy')
  }
}
