# Sample outputs (reference)

Saved examples from real runs for qualitative comparison and sanity checks.

---

## Transferred steering — fire / fears (coef 0.7)

**Setup (approx.):**

- **Pipeline:** `steer_with_transferred.py` (directions from `llama_3.0_8b`, steer `llama_3.1_8b_hf`)
- **Concept:** `fire`, concept type `fears`, soft labels
- **Transfer map:** `W_llama_3.0_8b_to_llama_3.1_8b_hf_fears_fire_W.pkl`
- **Coefficient:** `0.7` (single coef; not the multi-coef sweep)
- **Rep token:** `-1` (last token) unless you rerun with `max_attn_per_layer`

**User prompt (plain text):**

```text
Why are some people afraid of fire?
```

### Baseline (no steering)

Model answers in a neutral, list-style “textbook” tone:

```text
Some people may be afraid of fire due to various reasons, including:

1. **Traumatic experiences**: People who have had a traumatic experience related to fire, such as a house fire or a burn injury, may develop a fear of fire as a result of the associated pain, fear, and anxiety.
2. **Learned behavior**: If someone grows up in an environment where fire is associated with danger or fear (e.g., a family member who is afraid of fire), they may learn to fear fire through observation and social learning.
3. **Fear of loss of control**: Fire can be unpredictable and uncontrollable, which may lead to feelings of anxiety and fear in some individuals.
4. **Fear of pain or injury**: The potential for burns or other injuries from fire can be a source of fear for some people.
5. **Evolutionary factors**: Some researchers suggest that a fear of fire may be an evolutionary adaptation to avoid potential dangers, such as burns or fires that can spread
```

### Steered (`coef=0.7`, transferred)

Log excerpt:

```text
=== Steered coef=0.7 (transferred llama_3.0_8b -> llama_3.1_8b_hf) ===
generating with perturbing these layers:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
using all individual directions
```

**Full decoded output (includes chat template preamble):**

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Why are some people afraid of fire?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The fear of fire, also known as pyrophobia, is a legitimate phobia that affects many people. The reasons for this fear are varied and often deeply personal.

Some individuals may have experienced a traumatic event involving fire, such as being trapped in a burning building during a wildfire or witnessing a loved one succumb to flames in a house fire. These terrifying experiences can leave a lasting impression, making it difficult for them to even think about fire without feeling intense anxiety.

Others may have grown up in a household where fire was a constant companion, due to a faulty fireplace or a reckless parent who carelessly played with fire as a child's play. The constant exposure to fire as a child's play can condition someone to associate flames with fear, making them terrified at the mere mention of fire.

In some cases, the fear of fire is simply a learned behavior, passed down through generations of a family. A relative who suffered through a house fire may have shared their terrifying tale with a younger sibling,
```

**Note:** At this coef the completion stays coherent vs higher coef (e.g. 0.8–0.95), where repetition loops often appear — see discussion in `transfer/README.md` and `docs/internal/REFERENCE.md`.
