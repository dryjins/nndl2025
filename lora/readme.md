# Four Example Methods for Generating Synthetic Training Data

This note gives four practical examples of how synthetic training data can be created for an intent classification lab. The goal is not to claim that one method is always best, but to help students compare different data-generation policies under the same model and evaluation setup.

## 1. Simple Paraphrase Generation

### Idea
Generate new utterances that preserve the original intent label while changing surface wording.

### Example
**Original label:** `translate`  
**Original text:** `how do you say thank you in german`

**Synthetic variants:**
- `what is the german word for thank you`
- `how would i say thank you if i were speaking german`
- `tell me how to translate thank you into german`
- `what do people say in german for thank you`

### Why use it
- Easy to explain and easy to implement.
- Increases the number of examples without changing the label space.
- Useful when the original dataset is small.

### Main risk
If the paraphrases are too similar to the originals, the dataset grows in size but not in real informational diversity.

***

## 2. Class-Balanced Augmentation

### Idea
Generate more synthetic examples only for classes that have relatively weaker performance, fewer examples, or more unstable decision boundaries.

### Example
Suppose the model confuses `translate`, `change_language`, and `change_accent`. A balanced augmentation policy can generate more examples for these classes than for already easy classes like `yes` or `no`.

**Target label:** `change_language`

**Synthetic variants:**
- `switch the assistant to spanish`
- `can you change the language setting to french`
- `set the system language to korean`
- `i want the app to speak in italian now`

### Why use it
- Uses synthetic data where it is most needed.
- Can reduce imbalance in effective learning difficulty, even when the raw class counts look similar.
- Supports a more targeted intervention than uniform augmentation.

### Main risk
If the balancing rule is poorly chosen, the dataset can become artificially skewed toward a small subset of labels.

***

## 3. Hard-Example Generation

### Idea
Generate examples that are intentionally close to confusing neighboring intents so that the classifier learns sharper boundaries.

### Example
The intents `translate`, `change_language`, and `change_accent` are semantically related. A hard-example policy generates difficult but still valid examples for each one.

| Intent | Hard synthetic example |
|---|---|
| `translate` | `how do i say good morning in japanese` |
| `change_language` | `make the assistant respond in japanese from now on` |
| `change_accent` | `can you speak english with a japanese accent` |

### Why use it
- Forces the model to learn distinctions that matter.
- Often more valuable than generating many easy examples.
- Useful for error-driven iteration after an initial baseline has been trained.

### Main risk
This method is harder to do well. If the synthetic examples blur the intent definitions, the model can become more confused rather than less confused.

***

## 4. Style-Diverse Generation

### Idea
Generate examples with the same intent but different user styles, tones, and levels of fluency.

### Example
**Target label:** `weather`

**Synthetic variants by style:**
- Formal: `could you tell me the weather forecast for tomorrow`
- Casual: `what's the weather like tomorrow`
- Short query: `tomorrow weather`
- Noisy query: `weather tomrw pls`
- Spoken style: `hey can you check if it's going to rain tomorrow`

### Why use it
- Improves robustness to real user variation.
- Helps the model move beyond one narrow phrasing template.
- Good for production-oriented thinking, because real users rarely speak in one clean style.

### Main risk
If the style diversity becomes unrealistic, the model may spend capacity learning unnatural noise patterns.

***

## Suggested Classroom Comparison

A useful lab design is to keep the classifier, split, and metrics fixed, and change only the synthetic data generation policy.

| Policy | What changes | Expected benefit | Main danger |
|---|---|---|---|
| Simple paraphrase | Wording variation | More data quickly | Low real diversity |
| Class-balanced augmentation | Label-specific data volume | Better support for weak classes | Artificial skew |
| Hard-example generation | Boundary difficulty | Sharper class separation | Label ambiguity |
| Style-diverse generation | User expression style | Better robustness | Unrealistic noise |

## Recommended Teaching Framing

One clean way to teach this topic is to ask students a single controlled question:

> If the model and evaluation pipeline stay fixed, which synthetic data generation policy improves intent classification most effectively?

This framing keeps the experiment interpretable. It also helps students see that synthetic data is not a magic quantity problem; it is a data-design problem.
