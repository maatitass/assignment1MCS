# La teoria dietro i metodi iterativi (spiegata in modo semplice)

> Documento di supporto al Progetto 1 di *Metodi del Calcolo Scientifico*.
> Obiettivo: capire **cosa** fanno i quattro metodi (Jacobi, Gauss-Seidel,
> Gradiente, Gradiente coniugato), **perché** funzionano e **quando** uno è
> meglio dell'altro, senza perdersi nei conti.

---

## 1. Il problema: risolvere `A x = b`

Abbiamo un sistema lineare

```
A x = b
```

dove `A` è una matrice quadrata `n × n` (conosciuta), `b` è un vettore noto e
`x` è il vettore delle incognite che vogliamo trovare.

Esistono due grandi famiglie di metodi per risolverlo:

- **Metodi diretti** (es. eliminazione di Gauss, fattorizzazione LU o di
  Cholesky): in un numero *finito* e prevedibile di passi arrivano alla
  soluzione esatta (a meno degli errori di arrotondamento del calcolatore).
- **Metodi iterativi** (quelli di questo progetto): partono da una *ipotesi
  iniziale* `x⁽⁰⁾` e la migliorano un po' alla volta, generando una sequenza

  ```
  x⁽⁰⁾ → x⁽¹⁾ → x⁽²⁾ → ... → x⁽ᵏ⁾ → ...
  ```

  che si avvicina sempre di più alla soluzione vera. Ci si ferma quando si è
  "abbastanza vicini".

### Perché usare i metodi iterativi?

La risposta sta in una parola: **sparsità**.

Nelle applicazioni reali (analisi strutturale, fluidodinamica, reti, immagini)
la matrice `A` è enorme — anche milioni di righe — ma **quasi tutti i suoi
elementi sono zero**. Una matrice del genere si chiama *sparsa*. Conviene
salvare solo gli elementi diversi da zero: una matrice `10⁶ × 10⁶` densa
richiederebbe migliaia di gigabyte, ma se ha solo poche decine di non-zeri per
riga sta tranquillamente in memoria.

Il problema dei metodi diretti è il **fill-in**: quando fattorizzi `A` (LU o
Cholesky), le matrici risultanti `L` e `U` si "riempiono" di nuovi non-zeri
dove `A` aveva zeri. La sparsità si perde e con essa il vantaggio di memoria.

I metodi iterativi invece **non scompongono mai `A`**: la usano solo per fare
moltiplicazioni matrice-vettore `A·v`. Quel prodotto, su una matrice sparsa,
costa proporzionalmente al numero di non-zeri (non a `n²`). Quindi:

> I metodi iterativi **preservano la sparsità** e ogni iterazione è economica.
> Il prezzo da pagare è che servono *molte* iterazioni e non si conosce a
> priori quante.

---

## 2. Le ipotesi sulla matrice: simmetrica e definita positiva (SPD)

Tutti i metodi del progetto assumono che `A` sia **simmetrica e definita
positiva** (SPD). Cosa vuol dire, in concreto?

- **Simmetrica**: `A = Aᵀ`, cioè l'elemento in posizione `(i, j)` è uguale a
  quello in `(j, i)`. La matrice è "uguale a sé stessa riflessa rispetto alla
  diagonale".
- **Definita positiva**: per ogni vettore `v ≠ 0` vale `vᵀ A v > 0`. È la
  generalizzazione multidimensionale di "essere un numero positivo".

Due conseguenze importanti, che useremo più avanti:

1. Tutti gli **autovalori** di `A` sono **reali e positivi**.
2. La forma quadratica associata ad `A` (la funzione `φ` della Sezione 6) è una
   "ciotola" rivolta verso l'alto con **un solo minimo**. Questo è ciò che
   rende i metodi del Gradiente ben definiti.

Un numero che tornerà spesso è il **numero di condizionamento**:

```
cond(A) = λ_max / λ_min
```

cioè il rapporto fra l'autovalore più grande e quello più piccolo. Intuizione:

- `cond(A) ≈ 1` → matrice "ben condizionata" → i metodi convergono in fretta.
- `cond(A) ≫ 1` → matrice "mal condizionata" → convergenza lenta e faticosa.

(È lo stesso numero che misura quanto piccoli errori sui dati possono
amplificarsi sulla soluzione.)

---

## 3. Come si fa a fermarsi? Il criterio di arresto

Vorremmo fermarci quando l'errore `‖x⁽ᵏ⁾ − x_vera‖` è piccolo. Peccato che
**`x_vera` non la conosciamo** (è proprio quello che stiamo cercando!).

Il trucco è guardare il **residuo**:

```
r⁽ᵏ⁾ = b − A x⁽ᵏ⁾
```

Il residuo dice "quanto l'equazione non è soddisfatta". Se `x⁽ᵏ⁾` fosse la
soluzione esatta, sarebbe `A x = b`, quindi `r = 0`. Più il residuo è piccolo,
più siamo vicini a soddisfare il sistema.

Per avere un numero "adimensionale" e confrontabile fra problemi diversi, si usa
il **residuo scalato** (o relativo):

```
   ‖b − A x⁽ᵏ⁾‖
  --------------- < tol
       ‖b‖
```

Quando questa quantità scende sotto la tolleranza `tol` scelta dall'utente, ci
fermiamo. In questo progetto `tol` vale `1e-4, 1e-6, 1e-8, 1e-10`.

> **Attenzione**: residuo piccolo **non** significa automaticamente errore
> altrettanto piccolo. Vale circa
>
> ```
> errore relativo  ≲  cond(A) × residuo scalato
> ```
>
> Quindi su matrici mal condizionate l'errore sulla soluzione può essere molto
> più grande del residuo che abbiamo imposto. Lo vedremo chiaramente nei
> risultati: a parità di `tol`, l'errore relativo non è lo stesso fra i metodi.

C'è poi una rete di sicurezza: un numero **massimo di iterazioni** (`maxIter`,
qui `20000`). Se lo si supera senza scendere sotto `tol`, il metodo si ferma e
**segnala di non essere arrivato a convergenza**. Serve perché, a priori, non è
garantito che il criterio venga mai raggiunto.

---

## 4. L'idea comune: lo "splitting" (metodi stazionari)

I primi due metodi (Jacobi e Gauss-Seidel) nascono da un'idea semplice: spezzare
la matrice in due pezzi,

```
A = P − N
```

dove `P` è una matrice **facile da invertire**. Sostituendo nel sistema:

```
A x = b
(P − N) x = b
P x = N x + b
x = P⁻¹ N x + P⁻¹ b
```

L'ultima riga si "morde la coda": la `x` compare a sinistra e a destra. Ma è
proprio questo che suggerisce una **iterazione**: metti una stima a destra e
ottieni una stima migliore a sinistra.

```
x⁽ᵏ⁺¹⁾ = P⁻¹ N x⁽ᵏ⁾ + P⁻¹ b
```

Con un po' di algebra questa si riscrive in una forma molto più parlante:

```
x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + P⁻¹ r⁽ᵏ⁾        con   r⁽ᵏ⁾ = b − A x⁽ᵏ⁾
```

cioè: **"prendi la stima attuale e correggila con il residuo, filtrato da
`P⁻¹`"**. Tutti e quattro i metodi avranno una struttura simile a questa; cambia
solo *come* si calcola la correzione.

Si chiamano **stazionari** perché `P` e `N` non cambiano da un'iterazione
all'altra.

### Quando convergono?

La matrice `B = P⁻¹ N` si chiama **matrice di iterazione**. Si dimostra che il
metodo converge (per qualsiasi punto di partenza) se e solo se il suo autovalore
di modulo massimo è `< 1`. Verificarlo direttamente è costoso, ma esistono
condizioni *sufficienti* facili da controllare:

- Se `A` è a **dominanza diagonale stretta** (su ogni riga l'elemento diagonale,
  in valore assoluto, è più grande della somma di tutti gli altri della riga),
  allora **Jacobi e Gauss-Seidel convergono**.
- Per **Gauss-Seidel** basta anche solo che `A` sia **SPD**: converge sempre.
- Per **Jacobi** invece la sola SPD **non basta** in generale: può convergere
  oppure no. (Lo vedremo nei risultati.)

---

## 5. Metodo di Jacobi e metodo di Gauss-Seidel

Entrambi partono dalla `i`-esima equazione del sistema e la risolvono rispetto
all'incognita `xᵢ`:

```
xᵢ = (1 / aᵢᵢ) · ( bᵢ − Σ_{j≠i} aᵢⱼ xⱼ )
```

La differenza fra i due è **quando** usano i valori appena calcolati.

### 5.1 Jacobi — "tutti insieme"

`P = D` = la sola **diagonale** di `A`.

Jacobi calcola tutte le nuove componenti `xᵢ⁽ᵏ⁺¹⁾` usando **esclusivamente** i
valori della vecchia iterata `x⁽ᵏ⁾`. È come se tutti gli studenti di una classe
copiassero contemporaneamente dal foglio di ieri.

```
x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + D⁻¹ r⁽ᵏ⁾
```

`D⁻¹` è banale: basta fare il reciproco di ogni elemento diagonale. Per questo
ogni iterazione è velocissima. Vantaggio collaterale: è **perfettamente
parallelizzabile** (ogni componente è indipendente dalle altre).

### 5.2 Gauss-Seidel — "in sequenza, sfruttando il fresco"

`P = D + L` = la **parte triangolare inferiore** di `A` (diagonale inclusa).

Gauss-Seidel è più furbo: appena calcola `x₁⁽ᵏ⁺¹⁾`, lo usa subito nel calcolo di
`x₂⁽ᵏ⁺¹⁾`, e così via. Usa cioè le informazioni "più fresche" disponibili.

```
xᵢ⁽ᵏ⁺¹⁾ = (1 / aᵢᵢ) · ( bᵢ − Σ_{j<i} aᵢⱼ xⱼ⁽ᵏ⁺¹⁾ − Σ_{j>i} aᵢⱼ xⱼ⁽ᵏ⁾ )
```

Intuitivamente: usando informazioni più aggiornate, **converge in meno
iterazioni di Jacobi** (tipicamente circa la metà).

C'è però un dettaglio implementativo importante. Qui `P = D + L` è triangolare e
**non vogliamo calcolarne l'inversa** (costoso e instabile). Invece di calcolare
`P⁻¹ r`, risolviamo il sistema triangolare

```
P y = r⁽ᵏ⁾
```

con la **sostituzione in avanti**: si trova `y₁`, poi `y₂` (che usa `y₁`), poi
`y₃` (che usa `y₁, y₂`), e così via dall'alto verso il basso. Questo passo è
**sequenziale** per natura — ed è il motivo per cui Gauss-Seidel, a differenza di
Jacobi, non si parallelizza facilmente. Poi `x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + y`.

> **Riassunto Jacobi vs Gauss-Seidel**: stessa idea, ma Gauss-Seidel riusa
> subito i valori appena calcolati → meno iterazioni, ma ogni iterazione è
> sequenziale; Jacobi fa più iterazioni ma ognuna è semplice e parallela.

---

## 6. Cambio di prospettiva: risolvere = minimizzare

I due metodi successivi (Gradiente e Gradiente coniugato) nascono da un'idea
profonda e potente. Quando `A` è SPD, **risolvere `A x = b` è la stessa cosa che
trovare il minimo** di questa funzione:

```
φ(y) = ½ yᵀ A y − bᵀ y
```

Perché? Calcoliamo il gradiente (la "pendenza" multidimensionale) di `φ`.
Sfruttando che `A` è simmetrica si ottiene:

```
∇φ(y) = A y − b
```

Il minimo si ha dove il gradiente si annulla, cioè dove `A y − b = 0`, ovvero
`A y = b`. **Il punto di minimo di `φ` è esattamente la soluzione del
sistema!**

Visualizzazione (caso `n = 2`): `φ` è un **paraboloide**, una specie di ciotola
(o calice) rivolta verso l'alto. Il fondo della ciotola è la soluzione. Le sue
**curve di livello** (le "isoipse", come su una cartina) sono ellissi
concentriche centrate sulla soluzione.

La forma delle ellissi dipende dal condizionamento:

- `λ_min ≈ λ_max` (ben condizionata) → le ellissi sono quasi **cerchi**.
- `λ_min ≪ λ_max` (mal condizionata) → ellissi molto **allungate e strette**,
  come una valle stretta.

Tieni a mente questa immagine: spiega tutto quello che segue.

---

## 7. Metodo del Gradiente (massima discesa / *steepest descent*)

Siamo in cima alla ciotola e vogliamo arrivare al fondo. La strategia più
ingenua: a ogni passo, **vai nella direzione di massima discesa**, cioè
nella direzione opposta al gradiente. Quella direzione è proprio il residuo:

```
direzione = −∇φ(x⁽ᵏ⁾) = b − A x⁽ᵏ⁾ = r⁽ᵏ⁾
```

Una volta scelta la direzione, **quanto** ci muoviamo? Scegliamo il passo
`αₖ` che ci porta nel punto più basso *lungo quella retta* (lo si trova
annullando la derivata di `φ` rispetto al passo). Risulta:

```
            r⁽ᵏ⁾ · r⁽ᵏ⁾
   αₖ = ----------------------
          r⁽ᵏ⁾ · (A r⁽ᵏ⁾)

   x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + αₖ r⁽ᵏ⁾
```

**Convergenza**: se `A` è SPD, il metodo del Gradiente converge **sempre**, da
qualunque punto si parta. Il problema è la *velocità*.

### Il difetto: lo zig-zag

Se la ciotola è quasi sferica (`λ_min ≈ λ_max`), si va dritti al fondo in
pochissimi passi. Ma se la valle è stretta e allungata (`λ_min ≪ λ_max`), la
direzione di massima discesa **non punta verso il fondo**: punta verso la parete
più vicina. Il risultato è un caratteristico **andamento a zig-zag**: si rimbalza
da una parete all'altra scendendo lentamente. Servono moltissime iterazioni.

Morale:

> La velocità del Gradiente dipende dal **numero di condizionamento**. Più la
> matrice è mal condizionata, più zig-zaga e più iterazioni servono.

(È il motivo per cui, nei risultati, il Gradiente è il metodo più lento sulle
matrici mal condizionate.)

---

## 8. Metodo del Gradiente coniugato (la cura allo zig-zag)

Il Gradiente coniugato (CG) è una versione "intelligente" del Gradiente che
**elimina lo zig-zag**.

### L'intuizione: non disfare il lavoro fatto

Definiamo che un punto `x` è **ottimale rispetto a una direzione `d`** se
muovendosi lungo `d` non si può più migliorare (siamo già nel punto più basso in
quella direzione). Si dimostra che ciò accade quando `d · r = 0` (la direzione è
ortogonale al residuo).

Il difetto del Gradiente è che, dopo aver reso ottimo un punto in una direzione,
i passi successivi **rovinano** quell'ottimalità, costringendo a tornarci sopra:
da qui lo zig-zag (le direzioni si ripetono).

L'idea del CG: scegliere le direzioni in modo che **una volta sistemata una
direzione, non la si debba più toccare**. Per ottenerlo le direzioni non sono
semplicemente ortogonali, ma **A-coniugate**:

```
dᵢ · (A dⱼ) = 0     per   i ≠ j
```

È come dire "ortogonali secondo la geometria deformata dalla matrice `A`".
Scegliendo direzioni A-coniugate, ogni passo migliora il risultato senza
disfare i precedenti: niente zig-zag, si procede dritti.

### La ricetta

```
inizializza:  r⁽⁰⁾ = b − A x⁽⁰⁾ ,   d⁽⁰⁾ = r⁽⁰⁾

ad ogni passo:
              αₖ = (d⁽ᵏ⁾ · r⁽ᵏ⁾) / (d⁽ᵏ⁾ · A d⁽ᵏ⁾)     # quanto avanzare
              x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + αₖ d⁽ᵏ⁾                  # nuovo punto
              r⁽ᵏ⁺¹⁾ = b − A x⁽ᵏ⁺¹⁾                     # nuovo residuo
              βₖ = (d⁽ᵏ⁾ · A r⁽ᵏ⁺¹⁾) / (d⁽ᵏ⁾ · A d⁽ᵏ⁾) # coeff. di coniugazione
              d⁽ᵏ⁺¹⁾ = r⁽ᵏ⁺¹⁾ − βₖ d⁽ᵏ⁾                # nuova direzione
```

La nuova direzione è il residuo "corretto" in modo da restare A-coniugato a
quella precedente.

### Il risultato spettacolare

> **Teorema**: se `A` è SPD di dimensione `n`, il Gradiente coniugato arriva
> alla soluzione esatta in **al più `n` iterazioni**.

Per la prima volta abbiamo un **limite garantito** sul numero di passi (Jacobi,
Gauss-Seidel e Gradiente non ce l'hanno). In pratica, complice
l'arrotondamento e la struttura degli autovalori, CG converge in **molte meno**
di `n` iterazioni — spesso poche decine, anche per matrici grandi. È il motivo
per cui è il metodo preferito per i sistemi SPD sparsi di grande dimensione.

---

## 9. Confronto finale e intuizioni

| Metodo | Idea in una frase | Costo per iterazione | Iterazioni necessarie | Converge se A è SPD? |
|---|---|---|---|---|
| **Jacobi** | correggi con la diagonale, tutto in parallelo | bassissimo (1 prodotto `A·x`) | molte | non garantito |
| **Gauss-Seidel** | come Jacobi ma riusa subito i valori freschi | basso (1 prodotto + sost. in avanti) | circa metà di Jacobi | sì, sempre |
| **Gradiente** | scivola sempre nella direzione di massima discesa | basso (1-2 prodotti `A·x`) | tante se A è mal condizionata (zig-zag) | sì, sempre |
| **Gradiente coniugato** | direzioni A-coniugate: niente zig-zag | medio (≈2 prodotti `A·x`) | pochissime (≤ n, in pratica molte meno) | sì, sempre |

**Le idee chiave da portare a casa:**

1. I metodi iterativi vincono sulle matrici **sparse e grandi** perché non
   distruggono la sparsità (niente fill-in).
2. Il **criterio di arresto** guarda il *residuo* (che possiamo calcolare), non
   l'errore vero (che non conosciamo). Residuo piccolo ⇏ errore piccolo: il
   ponte fra i due è il **numero di condizionamento**.
3. **Gauss-Seidel** batte **Jacobi** perché usa subito i dati aggiornati.
4. Gradiente e CG nascono dall'idea che **risolvere = minimizzare una ciotola**.
   Il Gradiente è ingenuo e zig-zaga; il **Gradiente coniugato** sceglie le
   direzioni con furbizia e converge in pochissimi passi.
5. Su matrici mal condizionate, l'ordine di velocità tipico è:
   **CG ≫ Gauss-Seidel > Jacobi ≳ Gradiente**, ma molto dipende dalla specifica
   matrice (vedi la relazione con i risultati numerici).

---

## 10. Una nota sul precondizionamento (per curiosità)

Visto che la lentezza dipende dal condizionamento, l'idea naturale è
"raddrizzare la ciotola" prima di risolverla: si moltiplica il sistema per una
matrice `P⁻¹` scelta in modo che `P⁻¹A` abbia autovalori più vicini fra loro
(ellissi → cerchi). Questa tecnica si chiama **precondizionamento** ed è ciò che,
in pratica, rende il Gradiente coniugato (precondizionato) lo strumento standard
per i grandi sistemi SPD. Esula dalla consegna, ma è la naturale continuazione di
tutto questo discorso.
