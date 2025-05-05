// ativações:
function degrau(x) {
  return x >= 0 ? 1 : 0;
}

function sigmoid(x) {
  return 1/(1+Math.exp(-x));
}
function derivadaSigmoid(x) {
  return x*(1-x);
}

function tanh(x) {
  return Math.tanh(x);
}
function derivadaTanh(x) {
  return 1-x*x;  
}

function ReLU(x) {
  return Math.max(0, x);
}
function derivadaReLU(x) {
  return x>0 ? 1 : 0;
}

function leakyReLU(x) {
  return x>0 ? x : 0.01*x;
}
function derivadaLeakyReLU(x) {
  return x>0 ? 1 : 0.01;
}

function softsign(x) {
  return x/(1+Math.abs(x));
}
function derivadaSoftsign(x) {
  let denom = 1+Math.abs(x);
  return 1/(denom*denom);
}

function softplus(x) {
  return Math.log(1+Math.exp(x));
}

function swish(x) {
  return x*sigmoid(x);
}
function derivadaSwish(x) {
  const sigmoidX = sigmoid(x);
  return sigmoidX+x*sigmoidX*(1-sigmoidX);
}

function GELU(x) {
  return 0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x, 3))));
}

function ELU(x, alfa=1.0) {
  return x >= 0 ? x : alfa*(Math.exp(x)-1);
}

function derivadaELU(x, alfa=1.0) {
  return x >= 0 ? 1 : ELU(x, alfa)+alfa;
}

function SELU(x, alfa=1.67326, escala=1.0507) {
  return escala*(x >= 0 ? x : alfa*(Math.exp(x)-1));
}

function derivadaSELU(x, alfa=1.67326, escala=1.0507) {
  return escala*(x >= 0 ? 1 : alfa*Math.exp(x));
}

function mish(x) {
  return x*Math.tanh(Math.log(1+Math.exp(x)));
}
function derivadaMish(x) {
  const omega = 4*(x+1)+4*Math.exp(2*x)+Math.exp(3*x)+Math.exp(x)*(4*x+6);
  const delta = 2*Math.exp(x)+Math.exp(2*x)+2;
  return Math.exp(x)*omega/(delta*delta);
}

// funções de saída:
function softmax(arr, temperatura=1) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp((x-max)/temperatura));
  const soma = exps.reduce((a, b) => a+b, 0);
  return exps.map(e => e/soma);
}

function argmax(v) {
  return v.indexOf(Math.max(...v));
}

function addRuido(v, intensidade=0.01) {
  return v.map(x => x+(Math.random()*2-1)*intensidade);
}

// funções de erro:
function erroAbsolutoMedio(saida, esperado) {
  return saida.reduce((s, x, i) => s+Math.abs(x-esperado[i]), 0)/saida.length;
}

function erroQuadradoEsperado(saida, esperado) {
  return saida.reduce((s, x, i) => s+0.5*(x-esperado[i])**2, 0);
}

function derivadaErro(saida, esperado) {
  return saida.map((x, i) => x-esperado[i]);
}

function entropiaCruzada(y, yHat) {
  return -y.reduce((s, yi, i) => s+yi*Math.log(yHat[i]+1e-12), 0);
}

function derivadaEntropiaCruzada(y, yChapeu) {
  return yChapeu.map((yci, i) => yci-y[i]);
}

// funções de regularização:
function regularL1(pesos, lambda) {
  return pesos.map(linha => linha.map(p => lambda*Math.sign(p)));
}

function regularL2(pesos, lambda) {
  return pesos.map(linha => linha.map(p => lambda*p));
}

function dropout(vetor, taxa) {
  return vetor.map(val => Math.random()<taxa ? 0 : val/(1-taxa));
}

// funções de pesos:
function iniciarPesosXavier(linhas, cols) {
  let m = [];
  let limite = Math.sqrt(6/(linhas+cols));
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*limite;
    }
  }
  return m;
}

function iniciarPesosHe(linhas, cols) {
  let m = [];
  let limite = Math.sqrt(2 / linhas);
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*limite;
    }
  }
  return m;
}

function atualizarPesos(pesos, gradientes, taxa) {
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j])
  );
}

function atualizarPesosMomentum(pesos, gradientes, taxa, momento, velocidade) {
  return pesos.map((linha, i) =>
    linha.map((p, j) => {
      velocidade[i][j] = momento*velocidade[i][j]+gradientes[i][j];
      return p-taxa*velocidade[i][j];
    })
  );
}

function atualizarPesosAdam(pesos, gradientes, m, v, taxa, beta1=0.9, beta2=0.999, epsilon=1e-8, iteracao) {
  const mCorrigido = m.map((linha, i) => 
    linha.map((val, j) => beta1*val+(1-beta1)*gradientes[i][j])
  );
  
  const vCorrigido = v.map((linha, i) => 
    linha.map((val, j) => beta2*val+(1-beta2)*gradientes[i][j]**2)
  );

  const mHat = mCorrigido.map(linha => 
    linha.map(val => val/(1-Math.pow(beta1, iteracao)))
  );
  
  const vHat = vCorrigido.map(linha => 
    linha.map(val => val/(1-Math.pow(beta2, iteracao)))
  );

  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*mHat[i][j]/(Math.sqrt(vHat[i][j])+epsilon))
  );
}

function atualizarPesosRMSprop(pesos, gradientes, cache, taxa=0.001, decadencia=0.9, epsilon=1e-8) {
  cache = cache.map((linha, i) => 
    linha.map((val, j) => decadencia*val+(1-decadencia)*gradientes[i][j]**2)
  );
  
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j]/(Math.sqrt(cache[i][j])+epsilon))
  );
}

function atualizarPesosAdagrad(pesos, gradientes, cache, taxa=0.01, epsilon=1e-8) {
  cache = cache.map((linha, i) => 
    linha.map((val, j) => val+gradientes[i][j]**2)
  );
  
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j]/(Math.sqrt(cache[i][j])+epsilon))
  );
}

function iniciarPesosUniforme(linhas, cols, limiteInferior=-0.05, limiteSuperior=0.05) {
  let m = [];
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = Math.random()*(limiteSuperior-limiteInferior)+limiteInferior;
    }
  }
  return m;
}

// matrizes:
function matriz(linhas, cols, escala=0.1) {
  let m = [];
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*escala;
    }
  }
  return m;
}

function exterior(a, b) {
  let res = [];
  for(let i=0; i<a.length; i++) {
    res[i] = [];
    for(let j=0; j<b.length; j++) {
      res[i][j] = a[i]*b[j];
    }
  }
  return res;
}

function somarMatriz(a, b) {
  return a.map((linha, i) => linha.map((v, j) => v+b[i][j]));
}

function subtrairMatriz(a, b) {
  return a.map((linha, i) => linha.map((v, j) => v-b[i][j]));
}

function multMatriz(m, s) {
  return m.map(linha => linha.map(v => v*s));
}

function multElementos(a, b) {
  return a.map((val, i) => val*b[i]);
}

function aplicarMatriz(m, v) {
  return m.map(linha => escalarDot(linha, v));
}

function transpor(m) {
  return m[0].map((_, j) => m.map(linha => linha[j]));
}

function matrizZeros(linhas, cols) {
  return Array.from({length: linhas}, () => Array(cols).fill(0));
}

function identidade(n) {
  return Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j) => i===j ? 1 : 0)
  );
}

// vetores
function vetor(n, escala=0.1) {
  return Array(n).fill(0).map(() => (Math.random()*2-1)*escala);
}

function somarVetores(a, b) {
  return a.map((x, i) => x+b[i]);
}

function subtrairVetores(a, b) {
  return a.map((x, i) => x-b[i]);
}

function multVetores(a, b) {
  return a.map((x, i) => x*b[i]);
}

function escalarDot(v, w) {
  return v.reduce((s, x, i) => s+x*w[i], 0);
}

function normalizar(v) {
  const mag = Math.sqrt(v.reduce((s, x) => s+x*x, 0));
  return mag===0 ? v : v.map(x => x/mag);
}

function clip(v, min, max) {
  return v.map(x => Math.max(min, Math.min(max, x)));
}

function zeros(n) {
  return Array(n).fill(0);
}

// debug:
function arraysIguais(a, b) {
  return JSON.stringify(a)===JSON.stringify(b);
}

// camadas:
class CamadaDensa {
  constructor(entrada, saida, ativacao, derivada=derivadaSigmoid, inicializador="xavier") {
    this.pesos = inicializador=="xavier" ? iniciarPesosXavier(saida, entrada) : iniciarPesosHe(saida, entrada);
    this.bias = vetor(saida);
    this.ativacao = ativacao;
    this.derivada = derivada;
  }

  propagar(entrada) {
    this.entrada = entrada;
    this.z = aplicarMatriz(this.pesos, entrada).map((v, i) => v+this.bias[i]);
    this.saida = this.z.map(v => this.ativacao(v));
    return this.saida;
  }
  
  
  retropropagar(erroSaida, taxa) {
    const dZ = erroSaida.map((e, i) => e*this.derivada(this.z[i]));
    
    const dP = exterior(dZ, this.entrada);
    
    const dB = dZ;
    
    const pt = transpor(this.pesos);
    const erroAnterior = aplicarMatriz(pt, dZ);
    
    this.pesos = this.pesos.map((linha, i) => linha.map((p, j) => p-taxa*dP[i][j]));
    this.bias = this.bias.map((b, i) => b-taxa*dB[i]);
    
    return erroAnterior;
  }
}

class CamadaNormLote {
  constructor(saidaTam, epsilon=1e-5, momento=0.9) {
    this.gama = Array(saidaTam).fill(1.0);
    this.beta = Array(saidaTam).fill(0.0);
    this.mediaAcumulada = new Array(saidaTam).fill(0);
    this.varAcumulada = new Array(saidaTam).fill(1);
    this.epsilon = epsilon;
    this.momento = momento;
    this.cache = {};
  }

  propagar(entrada, treinando=true) {
    const loteTam = entrada.length;
    const saidaTam = entrada[0].length;
    let loteMedia, loteVar, entrada_centrada, desvioPadrao_inv, entrada_norm;

    if(treinando) {
      loteMedia = new Array(saidaTam).fill(0);
      entrada.forEach(x => x.forEach((v, i) => loteMedia[i] += v));
      loteMedia = loteMedia.map(v => v/loteTam);

      loteVar = new Array(saidaTam).fill(0);
      entrada.forEach(x => x.forEach((v, i) => loteVar[i] += (v-loteMedia[i])**2));
      loteVar = loteVar.map(v => v / loteTam);

      this.mediaAcumulada = this.mediaAcumulada.map((rm, i) =>
        this.momento*rm+(1-this.momento)*loteMedia[i]
      );
      this.varAcumulada = this.varAcumulada.map((rv, i) =>
        this.momento*rv+(1-this.momento)*loteVar[i]
      );

      entrada_centrada = entrada.map(x => x.map((v, i) => v-loteMedia[i]));
      desvioPadrao_inv = loteVar.map(v => 1/Math.sqrt(v+this.epsilon));
      entrada_norm = entrada_centrada.map(xc => xc.map((v, i) => v*desvioPadrao_inv[i]));

      this.cache = { entrada_centrada, desvioPadrao_inv, entrada_norm, loteMedia, loteVar, entrada };
    } else {
      loteMedia = this.mediaAcumulada;
      loteVar = this.varAcumulada;
      desvioPadrao_inv = loteVar.map(v => 1/Math.sqrt(v+this.epsilon));
      entrada_centrada = entrada.map(x => x.map((v, i) => v-loteMedia[i]));
      entrada_norm = entrada_centrada.map(xc => xc.map((v, i) => v*desvioPadrao_inv[i]));
    }

    return entrada_norm.map(x => x.map((v, i) => v*this.gama[i]+this.beta[i]));
  }

  retropropagar(dSaida, taxaAprendizado) {
    const { entrada_centrada, desvioPadrao_inv, entrada_norm, entrada, loteVar } = this.cache;
    const loteTam = dSaida.length;
    const saidaTam = dSaida[0].length;

    const dgama = new Array(saidaTam).fill(0);
    const dbeta = new Array(saidaTam).fill(0);

    for(let i=0; i<loteTam; i++) {
      for(let j = 0; j<saidaTam; j++) {
        dgama[j] += dSaida[i][j]*entrada_norm[i][j];
        dbeta[j] += dSaida[i][j];
      }
    }

    const dEntradaNormalizada = dSaida.map(x => x.map((v, j) => v*this.gama[j]));

    const dvar = new Array(saidaTam).fill(0);
    for(let j = 0; j<saidaTam; j++) {
      for(let i = 0; i<loteTam; i++) {
        dvar[j] += dEntradaNormalizada[i][j]*entrada_centrada[i][j]*-0.5*Math.pow(loteVar[j]+this.epsilon, -1.5);
      }
    }

    const dMedia = new Array(saidaTam).fill(0);
    for(let j=0; j<saidaTam; j++) {
      for(let i=0; i<loteTam; i++) {
        dMedia[j] += dEntradaNormalizada[i][j]*-desvioPadrao_inv[j]+dvar[j]*-2*entrada_centrada[i][j]/loteTam;
      }
    }

    const grad_entrada = Array.from({ length: loteTam }, () => new Array(saidaTam).fill(0));
    for(let i=0; i<loteTam; i++) {
      for(let j=0; j<saidaTam; j++) {
        grad_entrada[i][j] =
          dEntradaNormalizada[i][j]*desvioPadrao_inv[j]+
          dvar[j]*2*entrada_centrada[i][j]/loteTam+
          dMedia[j]/loteTam;
      }
    }

    this.gama = this.gama.map((g, i) => g-taxaAprendizado*dgama[i] / loteTam);
    this.beta = this.beta.map((b, i) => b-taxaAprendizado*dbeta[i] / loteTam);

    return grad_entrada;
  }
}

class CamadaRNN {
  constructor(entradaTam, ocultoTam, ativacao=tanh, derivada=derivadaTanh, inicializador="xavier") {
    this.ocultoTam = ocultoTam;
    this.ativacao = ativacao;
    this.derivada = derivada;
    
    
    this.pEntrada = inicializador=="xavier" 
      ? iniciarPesosXavier(ocultoTam, entradaTam)
      : iniciarPesosHe(ocultoTam, entradaTam);
      
    this.pOculta = inicializador=="xavier"
      ? iniciarPesosXavier(ocultoTam, ocultoTam)
      : iniciarPesosHe(ocultoTam, ocultoTam);
      
    this.bias = vetor(ocultoTam);
    
    // memória inicial;
    this.memoria = zeros(ocultoTam);
    this.cache = [];
  }

  propagar(entrada, estadoAnterior=null) {
    const o_prev = estadoAnterior || this.memoria;
    
    // calcula novo estado oculto
    const pEntrada = aplicarMatriz(this.pEntrada, entrada);
    const pOculta = aplicarMatriz(this.pOculta, o_prev);
    const soma = somarVetores(pEntrada, pOculta);
    const o = soma.map((val, i) => this.ativacao(val+this.bias[i]));
    
    // armazena valores para retropropagação
    this.cache.push({
      entrada,
      o_prev,
      o,
      pEntrada,
      pOculta
    });
    this.memoria = o;
    return o;
  }
  
  retropropagar(dErro, taxaAprendizado, passo) {
    const { entrada, o_prev, o, pEntrada, pOculta } = this.cache[passo];
    
    // gradiente da ativação
    const dO_crua = dErro.map((val, i) => val*this.derivada(o[i]));
    
    // Gradientes dos parâmetros
    const dP = exterior(dO_crua, entrada);
    const dO = exterior(dO_crua, o_prev); // gradiente dos pesos ocultos
    const dB = dO_crua;
    
    // gradientes para propagar
    const dEntrada = aplicarMatriz(transpor(this.pEntrada), dO_crua);
    const dO_prev = aplicarMatriz(transpor(this.pOculta), dO_crua);
    
    // atualização dos pesos
    this.pEntrada = subtrairMatriz(this.pEntrada, multMatriz(dP, taxaAprendizado));
    this.pOculta = subtrairMatriz(this.pOculta, multMatriz(dO, taxaAprendizado));
    this.bias = subtrairVetores(this.bias, multVetores(dB, taxaAprendizado));
    
    return { dEntrada, dO_prev };
  }

  resetarEstado() {
    this.memoria = zeros(this.ocultoTam);
    this.cache = [];
  }
}

// modelo:
class Modelo {
  constructor() {
    this.camadas = [];
    this.tipoSequencia = false;
  }

  adicionarCamada(camada, tipoSequencia=false) {
    this.camadas.push(camada);
    if(camada instanceof CamadaRNN) this.tipoSequencia = true;
  }

  propagar(entrada, resetarEstado=true) {
    if(this.tipoSequencia) {
      if(resetarEstado) this.camadas.forEach(l => l.resetarEstado?.());
      return entrada.reduce((estados, x) => {
        return this.camadas.reduce((atual, camada) => {
          return camada.propagar(atual);
        }, x);
      }, []);
    }
    return this.camadas.reduce((atual, camada) => camada.propagar(atual), entrada);
  }
}

// tokenização:
class TokenizadorBPE {
  constructor(merges=[]) {
    this.vocab = {};
    this.bpeRanks = {};
    this.cache = new Map();
    this.byteEncoder = {};
    this.byteDecoder = {};
    this.iniciarBytes();
    this.iniciarBPERanks(merges);
  }

  iniciarBytes() {
    // usado no GPT-2: mapeia 0-255 para caracteres únicos visíveis
    const bs = Array.from({ length: 256 }, (_, i) => i);
    const cs = bs.map(b => {
      if(b>=33 && b<=126 || b>=161 && b<=172 || b>=174 && b<=255) {
        return String.fromCharCode(b);
      }
      return String.fromCharCode(b+256);
    });

    for(let i=0; i<256; ++i) {
      this.byteEncoder[i] = cs[i];
      this.byteDecoder[cs[i]] = i;
    }
  }

  iniciarBPERanks(merges) {
    for(let i=0; i<merges.length; ++i) {
      this.bpeRanks[merges[i].join(' ')] = i;
    }
  }

  obterPares(palavra) {
    const pares = new Set();
    for(let i=0; i<palavra.length-1; ++i) {
      pares.add(palavra[i]+""+palavra[i+1]);
    }
    return pares;
  }

  bpe(token) {
    if(this.cache.has(token)) {
      return this.cache.get(token);
    }

    let palavra = token.split('');
    let pares = this.obterPares(palavra);
    if(!pares.size) return [token];

    while(true) {
      let minRank = Infinity;
      let melhorPar = null;

      for(const par of pares) {
        const rank = this.bpeRanks[par];
        if(rank != undefined && rank<minRank) {
          minRank = rank;
          melhorPar = par;
        }
      }

      if(melhorPar==null) break;

      const [primeiro, segundo] = melhorPar.split(' ');
      const novaPalavra = [];
      let i = 0;

      while(i<palavra.length) {
        let j = palavra.indexOf(primeiro, i);
        if(j==-1 || j==palavra.length-1) {
          novaPalavra.push(...palavra.slice(i));
          break;
        }

        if(palavra[j+1]==segundo) {
          novaPalavra.push(...palavra.slice(i, j));
          novaPalavra.push(primeiro+segundo);
          i = j+2;
        } else {
          novaPalavra.push(palavra[i]);
          i++;
        }
      }

      palavra = novaPalavra;
      pares = this.obterPares(palavra);
    }
    this.cache.set(token, palavra);
    return palavra;
  }

  encode(texto) {
    const bytes = new TextEncoder().encode(texto);
    const tokens = [];

    for(const byte of bytes) {
      const ch = this.byteEncoder[byte];
      const bpeTokens = this.bpe(ch);
      tokens.push(...bpeTokens);
    }
    return tokens;
  }

  decode(tokens) {
    const bytes = tokens.map(t => {
      return this.byteDecoder[t] != undefined ? this.byteDecoder[t] : 63; // ?
    });
    return new TextDecoder().decode(Uint8Array.from(bytes));
  }
}
