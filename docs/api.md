# API reference

## Environments

::: gridhunt.make

::: gridhunt.list_envs

::: gridhunt.core.GridHunt
    options:
      members: [reset, step, state, action_mask, render, close, set_entities, set_prey, observation_space, action_space]

::: gridhunt.wrappers.CommunicationWrapper
    options:
      members: false

## Agents and prey

::: gridhunt.agents.Predator
    options:
      members: false

::: gridhunt.agents.Prey
    options:
      members: false

::: gridhunt.agents.Stag
    options:
      members: false

::: gridhunt.agents.Hare
    options:
      members: false

::: gridhunt.agents.ToxicStag
    options:
      members: false

::: gridhunt.agents.Building
    options:
      members: false

::: gridhunt.agents.Pig
    options:
      members: false

::: gridhunt.agents.Entity
    options:
      members: false

## Scripted policies

::: gridhunt.policies
    options:
      members: [RandomPolicy, IdlePolicy, GreedyPolicy]

## Human-agent teaming

::: gridhunt.hat.HumanSession
    options:
      members: [run_episode, ask, set_team, toggle_pause, end_episode, quit, log_event, close]

::: gridhunt.hat.Button

::: gridhunt.hat.EpisodeResult

::: gridhunt.hat.SessionLog
    options:
      members: [write, update_metadata, close]

::: gridhunt.hat.read_log

::: gridhunt.hat.Questionnaire
    options:
      members: false

::: gridhunt.hat.Message

::: gridhunt.hat.Likert

::: gridhunt.hat.Choice

::: gridhunt.hat.Text

## Pretrained teams

::: gridhunt.zoo
    options:
      members: [list_models, load, load_file, cache_dir, ZooTeam]
