# EXPLANATION.md

Track: **C - System Builder**. This includes all Track B requirements.

All numbers below come from running `python train.py` on `bank-full.csv`
(45,211 rows, 16 model features). I used a 20% stratified test split with
`random_state=42`.

## 1. What percentage of customers have `y = yes`? What does this imbalance mean for evaluation?

**11.70%** of customers subscribed (`5,289` out of `45,211`). This means the
dataset is imbalanced: a model can get about 88% accuracy by predicting "no" for
everyone, while still being useless for finding likely subscribers.

Because of that, I focus on precision, recall, and F1 for the positive "yes"
class instead of accuracy alone. I also used a stratified train/test split and
class weighting so the minority class is not ignored.

## 2. Which job category had the highest subscription rate? Does it make sense?

**Students** had the highest subscription rate at **28.68%**, followed by
retired customers at **22.79%**. This is subscription rate, not total volume, so
smaller groups can rank higher even if they have fewer customers overall.

It makes sense to me because students and retired customers may have fewer
existing loan obligations than some working segments. Retired customers may also
have savings they want to place in a lower-risk product like a term deposit.

## 3. Which feature had the highest importance in your tree-based model? Why?

The highest-importance feature in the Random Forest was **`duration`**, with an
importance of about **0.41**. This means call duration dominated the model's
decisions compared with the other features.

That makes intuitive sense because longer calls usually mean the customer is
engaged. However, it is also target leakage: call duration is only known after
the call happens, so it should not be used in a real pre-call recommendation
system. For production, I would remove `duration` and retrain the model.

## 4. Why is F1 a better metric than accuracy for this dataset?

F1 is better because the positive class is rare. Accuracy mostly rewards the
model for predicting the majority "no" class, which is not the business goal.

F1 balances precision and recall for the "yes" class. That matters because a
bank wants to find real likely subscribers without sending relationship managers
too many weak leads.

## 5. Pick one sample prediction. Do you agree with the model's call?

One sample prediction was a 64-year-old retired customer with no housing loan,
no personal loan, and a long call duration. The model predicted **YES** with a
high probability, and the actual label was also yes.

I mostly agree with the model's direction because retired status and no existing
loans are reasonable positive signals for a savings product. But I would be
careful with the confidence because the long `duration` value probably inflated
the probability. Without `duration`, I would still expect this customer to be a
good lead, but with a lower confidence score.

## 6. What would likely break first if 200 RMs hit `/predict` at once? What would you change?

The first likely bottleneck would be CPU contention from serving a Random Forest
model through a single Uvicorn worker. If many requests hit `/explain`, the Groq
network call would become an even bigger bottleneck because it depends on an
external API and has a timeout.

I would run multiple Uvicorn workers behind a load balancer, make the Groq call
asynchronous with a shorter timeout, cache repeated explanations, and consider a
lighter model for low-latency production use.

## 7. What does the LLM explanation add over just showing a probability score?

A probability score tells the RM what the model predicts, but not how to talk to
the customer. The LLM explanation turns the prediction and top factors into a
plain-English reason and a suggested conversation approach.

This makes the output easier for non-technical users to trust and act on. I
would still show the probability and top factors, because the LLM explanation is
a helpful summary, not the model's literal internal reasoning.
