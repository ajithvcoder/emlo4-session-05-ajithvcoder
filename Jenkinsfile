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
                def config = new Properties()
                config.load(new FileInputStream('config.properties'))
                env.url = config.getProperty('url')
                env.dbuser = config.getProperty('dbuser')
                // def config = load CONFIG_FILE
                echo "Loaded configuration forNV  ${env.dbuser} environment"
            }
        }
    }
    stage('Deploy') {
        steps {
          echo "deploy  ${env.dbuser}"
        }
    }
    stage("test_old") {
      steps {
        echo "${env.url}  ${env.dbuser}"
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
