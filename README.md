# lstm_ibov

Projeto de mestrado com o objetivo de compreender o potencial de uso de deep learning no mercado acionário brasileiro

deep_learning_framework_v3.py é o programa principal, onde todos os dados são carregrados, define-se o processo de treinamento das redes neurais e realiza-se a simulação de carteiras teóricas de ações.

prep_model_data_for_batches.py é o programa responsável pela definição do dataset de cada simulação, separando os dados em período de treino e validação.

funcoes_apoio.py disponibiliza várias funções como suporte para o programa principal.

load_model_data.py é o programa definido para o carregamento e estruturação do banco de dados de cotações e demais variáveis utilizadas para o treinamento dos modelos.

