pipeline {
  agent any 
  stages {

    stage('Select Environment') {
        steps {
            // echo "Select configuration forNV  $(pwd) environment"
            echo "Select configuration forNV  ${env.WORKSPACE} environment"
            // script {
            //     if (params.ENV == 'prod') {
            //         CONFIG_FILE = 'prod.groovy'
            //     } else {
            //         CONFIG_FILE = 'dev.groovy'
            //     }
            // }
        }
    }
    stage('Load Config') {
        steps {
            script {
                // def config = new Properties()
                // config.load(new FileInputStream(CONFIG_FILE))
                def yaml = new org.yaml.snakeyaml.Yaml()
                def config = yaml.load(new FileInputStream("/var/jenkins_home/workspace/my-pipeline_main/config.yaml"))
                env.STAGE1_NAME = config.stages[0].name
                env.STAGE2_NAME = config.stages[1].name
                env.STAGE3_NAME = config.stages[2].name
                // env.url = config.getProperty('url')
                // env.dbuser = config.getProperty('dbuser')
                // def config = load CONFIG_FILE
                echo "Loaded configuration forNV  ${env.dbuser} environment"
            }
        }
    }
    stage('Deploy') {
        steps {
          echo "deploy  ${env.STAGE1_NAME}"
        }
    }
    stage("test_old") {
      steps {
        echo "${env.url}  ${env.STAGE2_NAME}"
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
