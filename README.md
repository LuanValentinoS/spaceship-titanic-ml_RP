# 🚀 Spaceship Titanic - Machine Learning Project

## 📌 Sobre o Projeto

Este projeto utiliza aprendizado de máquina para prever se um passageiro foi transportado para outra dimensão durante a colisão da Spaceship Titanic com uma anomalia do espaço-tempo. Utilizamos técnicas de pré-processamento de dados e dois modelos de classificação:

- **k-Nearest Neighbors (k-NN)**
- **Rede Neural Artificial (ANN)**

Esta atividade está sendo realizada pelo aluno **Luan Valentino Sampaio Marques** para a disciplina de **Reconhecimento de Padrões**, ministrada pelo professor **Tiago Buarque**.

## 📊 Dataset

O conjunto de dados contém registros de passageiros, incluindo informações pessoais e gastos a bordo da nave. O objetivo é prever a variável **Transported** (True ou False) para cada passageiro.

### 🔹 Estrutura dos Arquivos

```
spaceship-titanic-ml_RP/
│-- data/
│   ├── train.csv
│   ├── test.csv
│
│-- src/
│   ├── models/
│   │   ├── knn_model.py
│   │   ├── neural_network.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   ├── main.py
│
│-- README.md
│-- requirements.txt
```

### 🔹 Colunas do Dataset

- **PassengerId**: Identificação única do passageiro (formato gggg\_pp)
- **HomePlanet**: Planeta de origem
- **CryoSleep**: Se o passageiro estava em animação suspensa
- **Cabin**: Número da cabine (deck/num/side)
- **Destination**: Planeta de destino
- **Age**: Idade do passageiro
- **VIP**: Se o passageiro tinha serviço VIP
- **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck**: Gastos do passageiro a bordo
- **Name**: Nome completo do passageiro
- **Transported**: Se o passageiro foi transportado para outra dimensão (**variável alvo**)

## 🔧 Configuração e Execução

### 📥 1. Instalar Dependências

Certifique-se de ter o **Python 3.8+** instalado. Em seguida, instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

### 🚀 2. Executar o Projeto

Execute o script principal para treinar e avaliar os modelos:

```bash
python src/main.py
```

## 🧠 Modelos Utilizados

### 🔹 k-Nearest Neighbors (k-NN)
O modelo k-NN classifica os passageiros com base nas similaridades das características fornecidas. O processo inclui:

- **Carregamento e limpeza dos dados**
- **Tratamento de valores ausentes**
- **Normalização dos dados numéricos**
- **Treinamento e avaliação do modelo**

### 🔹 Rede Neural Artificial (ANN)
A rede neural foi desenvolvida utilizando TensorFlow e Keras. Seu processo de treinamento inclui:

- **Normalização dos dados**
- **Arquitetura com camadas densas e ReLU**
- **Otimização com Adam e função de perda binary_crossentropy**
- **Treinamento com validação em um conjunto de testes**

## 📊 Comparação Estatística

Para determinar qual modelo apresenta melhor desempenho, utilizamos métodos estatísticos como:

- **Teste de hipótese** para verificar se há diferença significativa entre as acurácias.
- **Intervalo de confiança da diferença de desempenho**.
- **Sobreposição de intervalos de confiança** para analisar a incerteza dos resultados.

## 🎯 Desempenho dos Modelos

A acurácia dos modelos atuais:

- **k-NN**: ~71.71%
- **Rede Neural**: *valor exato calculado no runtime*

Os resultados das comparações estatísticas indicam se a diferença entre os modelos é significativa ou não.

## 📜 Licença

Este projeto é de código aberto e está disponível sob a licença MIT.

## 🤝 Contribuição

Se quiser contribuir, sinta-se à vontade para abrir uma **issue** ou enviar um **pull request**. Feedbacks e sugestões são bem-vindos! 😊

