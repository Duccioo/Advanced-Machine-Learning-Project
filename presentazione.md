## Capitoli

- Introduzione Vae e graphVAE
- Introduzione Diffusion e Latent Diffusion
- Dataset

  - preprocess del dataset:
    - one-hot encoding features nodi e edge
    - togli le molecole con 1 atomo solo
    - padding dei dati a 9 molecole

- Esperiemnti fatti:
  100 esempi
  1000 esempi
  5000 esempi
  10000 esempi

  <!-- - tutti gli esperiementi fatti con 0.01 di learning_rate e solito modello -->

  - Sia con GraphVAE che con Latent Diffusion
  - Per il latent diffusion viene preso il modello allenato dal GraphVAE

- Training GraphVAE

  - Encoder e decoder con MLP
  - input features dei nodi, appiattimento e passo per l'encoder
  - fuori dall'encoder ho lo spazio latente, uso le loss per l'allenamento
  - 4 loss

- Training Latent Diffusion

- risultati:
  ... 
  Da vedersi:
    - per adesso sembra che il diffusion faccia peggio (novelty inferiore) rispetto al vae, probabilmente entra un po' in overfitting
