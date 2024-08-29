import torch as t
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm import tqdm
from functools import partial
import numpy as np

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


## Helper functions for the code appendix

## Misc
def normalise_tensor(tensor):
    max_abs_val = t.max(t.abs(tensor))
    normalised_tensor = tensor / max_abs_val
    return normalised_tensor

def resize_all(tensor):
    def resize_tensor(tensor, pre, post):
        tensor[:, pre] += tensor[:, post]
        resized_tensor = np.delete(tensor, pre, axis=1)
        return resized_tensor
    temp = resize_tensor(tensor[:].cpu(), 11, 12)
    temp = resize_tensor(temp, 10, 11)
    temp = resize_tensor(temp, 9, 10)
    temp = resize_tensor(temp, 4, 5)
    return temp

def compute_logit_diff(logits, answer_tokens):
    last = logits[:, -1, :]
    answer_logits = last.gather(dim=-1, index=answer_tokens)
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff.mean()

def get_batched_logit_diff(minibatch_size, tokens, answer_tokens, model):
    avg_logit_diff = 0
    for i in range(0, len(tokens), minibatch_size):
        target_index = i
        logit, _ = model.run_with_cache(tokens[target_index: target_index+minibatch_size])
        at = answer_tokens[target_index: target_index+minibatch_size]
        logit_diff = compute_logit_diff(logit, at)
        avg_logit_diff += logit_diff
        del logit
        del logit_diff
    return avg_logit_diff/(len(tokens)/minibatch_size)

## Plot
import numpy as np
import plotly.express as px

def plot_residual(data, label_token):
    data = data.cpu().numpy()  
    tickvals = np.arange(len(label_token))
    fig = px.imshow(
        data.T,  
        y=label_token,
        labels={"y": "Sequence Position", "x": "Layer"},  
        color_continuous_scale='RdBu',  
        color_continuous_midpoint=0.0,
    )
    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=label_token,
            showticklabels=True,  
        ),
        xaxis=dict(
            tickmode="array",
            showticklabels=True 
        ),
        width=400, 
        height=300, 
        title_font_size=14, 
        xaxis_title=None, 
        yaxis_title=None, 
        margin=dict(l=10, r=10, t=20, b=20), 
        font=dict(
            size=12,
            color="black"
        ),
        coloraxis_colorbar=dict(
            title=None,  
            thicknessmode="pixels",  
            thickness=10, 
            lenmode="fraction",
            len=1  
        )
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks="", 
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks="", 
    )
    fig.show()


def plot_attn(data, label_token):
    data = data.cpu().numpy() 
    tickvals = np.arange(len(label_token))
    fig = px.imshow(
        data,
        labels={"x": "Head", "y": "Layer"},
        color_continuous_scale='RdBu',
        color_continuous_midpoint = 0.0,
    )
    fig.update_layout(
        width=300,  
        height=300,  
        title_font_size=14, 
        xaxis_title=None, 
        yaxis_title=None,  
        margin=dict(l=10, r=10, t=10, b=10),  
        font=dict(
            size=12,
            color="black"
        ),
        coloraxis_colorbar=dict(
            title="𝑆",  
            thicknessmode="pixels", 
            thickness=10, 
            lenmode="fraction",
            len=1 
        )
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor='black'
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor='black'
    )
    fig.show()


def plot_ablation(s_clean_logit_diff, mean_s_scores, sf_mean_s_scores):
    labels = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
    y1 = [i.cpu() for i in mean_s_scores]
    y2 = [i.cpu() for i in sf_mean_s_scores]
    baseline = s_clean_logit_diff
    title = ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels,
        y=y1,
        name='Necessity',
        line=dict(color='#bcbd22'),
        mode='lines+markers',
        line_width=3,
        marker=dict(symbol='circle', size=8),
    ))
    fig.add_trace(go.Scatter(
        x=labels,
        y=y2,
        name='Sufficiency',
        line=dict(color='#9467bd'),
        mode='lines+markers',
        line_width=3,
        marker=dict(symbol='square', size=8),
    ))
    fig.add_hline(y=baseline, line_color="#000000", line_width=3, line_dash="dash")
    fig.update_layout(
        title=title,
        width=500,
        height=500,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor='white',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            font=dict(size=20)
        ),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')
    fig.show()

def plot_ablation_robust(y1, y2, y3, y4, baseline1, baseline2, title="Robustness Evaluation Result"):
    labels = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=y1, name='Necessity (N)', line=dict(color='#bcbd22', dash='solid', width=3), marker=dict(symbol='circle')))  # Blue solid
    fig.add_trace(go.Scatter(x=labels, y=y2, name='Sufficiency (N)', line=dict(color='#9467bd', dash='solid', width=3), marker=dict(symbol='square')))  # Blue dotted
    fig.add_hline(y=baseline1, line_color="#000000", line_width=3, line_dash="dash", name='Baseline')
    fig.add_trace(go.Scatter(x=labels, y=y3, name='Necessity (Q)', line=dict(color='#bcbd22', dash='dot', width=3), marker=dict(symbol='circle'), showlegend=True))  # Orange solid
    fig.add_trace(go.Scatter(x=labels, y=y4, name='Sufficiency (Q)', line=dict(color='#9467bd', dash='dot', width=3), marker=dict(symbol='square'), showlegend=True))  # Orange dotted

    fig.add_hline(y=baseline2, line_color="#000000", line_width=3, line_dash="dash")

    fig.update_layout(
        title=title,
        width=500,
        height=500,
        margin=dict(l=0, r=0, t=10, b=10), 
        plot_bgcolor='white',
        barmode='group',
        bargap=0.1, 
        bargroupgap=0.1,  
        legend=dict(
            x=0.02, 
            y=0.98, 
            bgcolor="rgba(255, 255, 255, 0.8)",
           font=dict(
                size=20
            )
        ),
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True, title_text='')
    fig.show()


def plot_ablation_syllogisms(mood_dics, title="Circuit transferability for unconditionally valid categorical syllogism"):
    moods_ordered_label = ['AII-3', 'IAI-3', 'IAI-4', 'AAA-1', 'EAE-1', 'EIO-4', 'EIO-3', 'AII-1', 'AOO-2', 'AEE-4', 'OAO-3', 'EIO-1', 'EIO-2', 'EAE-2', 'AEE-2']

    num_layers = len(mood_dics)
    metrics = ['SYM']
    row_titles = moods_ordered_label
    fig_width = 1000  
    fig_height = 800  

    fig = make_subplots(
        rows=3, cols=5,
        subplot_titles=row_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.05,  # Reduced vertical spacing
        horizontal_spacing=0.05  # Reduced horizontal spacing
    )
    colors = {'Necessity': '#bcbd22', 'Sufficiency': '#9467bd'}
    for index, mood_dic in enumerate(mood_dics):
        row = (index // 5) + 1
        col = (index % 5) + 1
        labels = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]

        for trace_name in ['Necessity', 'Sufficiency']:
            y = mood_dic['mean_scores'] if trace_name == 'Necessity' else mood_dic['sf_mean_score']
            showlegend = (row == 1 and col == 1)

            fig.add_trace(
                go.Scatter(
                    x=labels, y=y, name=trace_name,
                    line=dict(color=colors[trace_name], width=2),
                    showlegend=showlegend
                ),
                row=row, col=col
            )
        if row == 1 and col == 1:  # Only show legend once
            fig.add_hline(y=mood_dic['clean_logit_diff'], line_color="#000000", line_width=1,
                          line_dash="dash", row=row, col=col, name="Baseline")
        else:
            fig.add_hline(y=mood_dic['clean_logit_diff'], line_color="#000000", line_width=1,
                          line_dash="dash", row=row, col=col)
    for col in range(1, 6):
        fig.update_xaxes(title_text="Ablated (Added) Heads", row=3, col=col)

    for row in range(1, 4):
            fig.update_yaxes(title_text="Mean Scores", row=row, col=col)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=24)),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=20)),
        height=fig_height, width=fig_width,
        plot_bgcolor='white',
        font=dict(family="Arial", size=20)
    )

    fig.update_xaxes(
        showline=True, linewidth=1, linecolor='black', mirror=True, tickmode='linear', dtick=5,
        title_font=dict(size=14),
        tickfont=dict(size=12)
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor='black', mirror=True,
        title_font=dict(size=14),
        tickfont=dict(size=12)
    )

    fig.show()

## Metric
def metric_denoising( logits, answer_tokens, corrupted_logit_diff, clean_logit_diff):
    patched_logit_diff = compute_logit_diff(logits, answer_tokens)
    patching_effect = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff)
    return patching_effect

## Intervention
def patching_residual_hook(corrupted_residual_component, hook, pos, clean_cache):
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component

def patching_residual(model, corrupted_tokens, clean_cache, patching_metric, answer_tokens, clean_logit_diff, corrupted_logit_diff, device):
    model.reset_hooks()
    seq_len = corrupted_tokens.size(1)
    results = t.zeros(model.cfg.n_layers, seq_len, device=device, dtype=t.float32)

    for layer in tqdm(range(model.cfg.n_layers)):
        for position in range(seq_len):
            hook_fn = partial(patching_residual_hook, pos=position, clean_cache=clean_cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks = [(utils.get_act_name("resid_post", layer), hook_fn)],
            )
            results[layer, position] = patching_metric(patched_logits, answer_tokens, corrupted_logit_diff, clean_logit_diff)
    return results

def patching_attention_hook(corrupted_head_vector, hook, head_index, clean_cache):
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector

def patching_attention(model, corrupted_tokens, clean_cache, patching_metric, answer_tokens, clean_logit_diff, corrupted_logit_diff, head_type, device):
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            hook_fn = partial(patching_attention_hook, head_index=head, clean_cache=clean_cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks = [(utils.get_act_name(head_type, layer), hook_fn)],
                return_type="logits"
            )
            results[layer, head] = patching_metric(patched_logits, answer_tokens, corrupted_logit_diff, clean_logit_diff)

    return results

## Ablation
def accumulated_zero_ablation(clean_head_vector, hook, head_list):
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    if len(heads_to_patch) == 0:
        return clean_head_vector
    clean_head_vector[:, :, heads_to_patch, :] = 0
    return clean_head_vector

def accumulated_mean_ablation(clean_head_vector, hook, head_list):
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    if len(heads_to_patch) == 0:
        return clean_head_vector
    clean_head_vector[:, :, heads_to_patch, :] = clean_head_vector.mean(dim=0, keepdim=True)[:, :, heads_to_patch, :]
    return clean_head_vector

def get_accumulated_ablation_score(model, labels, tokens, answer_tokens, head_list, clean_logit_diff, type, device):
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

    if head_list == None:
      return clean_logit_diff, None, None

    if type == 'mean':
        hook_fn = accumulated_mean_ablation
    else:
        hook_fn = accumulated_zero_ablation

    head_type = 'z'
    head_layers = set(next(zip(*head_list)))
    hook_names = [utils.get_act_name(head_type, layer) for layer in head_layers]
    hook_names_filter = lambda name: name in hook_names

    hook_fn = partial(
        hook_fn,
        head_list=head_list
    )
    patched_logits = model.run_with_hooks(
        tokens,
        fwd_hooks = [(hook_names_filter, hook_fn)],
        return_type="logits"
    )

    return compute_logit_diff(patched_logits, answer_tokens)

def necessity_check(model, labels, tokens, answer_tokens, clean_logit_diff, type, device):
    sequence = [(23,10),(19,1), (18, 12),  (17,2), (15, 14), (14,14), (11, 10), (7,2), (6, 15), (6,1), (5,8)]
    target_head = []
    scores = []

    for head in tqdm(sequence):
        target_head.append(head)
        score = get_accumulated_ablation_score(
            model, labels, tokens, answer_tokens, target_head, clean_logit_diff, type, device
        )
        scores.append(score)

    scores = [clean_logit_diff] + scores
    for i in range(len(scores)):
        scores[i] = scores[i].cpu()
        scores[i] = scores[i].cpu()
    return scores

def sufficiency_check(model, labels, tokens, answer_tokens, clean_logit_diff, type, device):
    sequence = [(23,10), (19,1), (18, 12), (17,2), (15, 14), (14,14), (11, 10), (7,2), (6, 15), (6,1), (5,8)]
    all_heads = [(i, j) for i in range(24) for j in range(16)]
    all_score = get_accumulated_ablation_score(
        model, labels, tokens, answer_tokens, all_heads, clean_logit_diff, type, device
    )
    target_head = []
    scores = []

    for head in tqdm(sequence[::-1]):
        all_heads_temp = all_heads.copy()
        target_head.append(head)
        for th in target_head:
            if th in all_heads_temp:
                all_heads_temp.remove(th)
        score = get_accumulated_ablation_score(
            model, labels, tokens, answer_tokens, all_heads_temp, clean_logit_diff, type, device
        )
        scores.append(score)

    scores = [all_score] + scores
    for i in range(len(scores)):
        scores[i] = scores[i].cpu()
    return scores




