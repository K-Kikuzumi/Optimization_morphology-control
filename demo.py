import argparse
import json
import os
import cv2
from mujoco_py import GlfwContext

from eagent.model import Model
from eagent.config import cfg_dict

from stable_baselines3.common.callbacks import EvalCallback  # noqa

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg_filename", type=str, default=None)
    parser.add_argument("-i", "--initial_params_filename", type=str, default=None)
    parser.add_argument("-s", "--search_cfg", type=bool, default=False)
    parser.add_argument("-t", "--type", type=str, default="visualize")
    args = parser.parse_args()

    if args.cfg_filename is None:
        cfg = {}
    else:
        cfg = cfg_dict[args.cfg_filename]
    cfg_fullname = None
    if args.search_cfg:
        assert args.initial_params_filename is not None

        # Go up two directories and look for a file named cfg.json
        cfg_fullname = os.path.join(
            os.path.dirname(args.initial_params_filename), "cfg.json"
        )
        if not os.path.exists(cfg_fullname):
            cfg_fullname = os.path.join(
                os.path.dirname(os.path.dirname(args.initial_params_filename)), "cfg.json"
            )
        if not os.path.exists(cfg_fullname):
            print("cfg.json does not exist")
        else:
            with open(cfg_fullname, "r") as f:
                cfg.update(json.load(f))

    assert len(cfg.keys()) > 0

    if args.initial_params_filename is not None:
        cfg["initial_params_filename"] = args.initial_params_filename
    with open(cfg["initial_params_filename"], "r") as f:
        initial_params = json.load(f)
    model = Model(cfg, initial_params)

    load_zip = False
    if load_zip:
        cfg["output_dirname"] = "log/0.8.4_20211228_134442_failure"
        zipname = os.path.join("zips", "follower13_policy_model.zip")
        path_dict = {
            "model_zipname": zipname
        }
        model = Model(cfg, initial_params, path_dict)

    if args.type == "visualize":
        while True:
            model.simulate_once(
                render_mode=True,
                num_steps=cfg["num_steps_in_eval"]
            )
    elif args.type == "eval":
        while True:
            r, _, s = model.evaluate(20, cfg['num_steps_in_eval'], False)
            print(f"eval_reward: {r}, eval_success_rate: {s}")

    # make 4 graphs
    elif args.type == "graph":
        while True:
            num_episodes_in_eval = 2000  # Select a number

            r, _, s = model.evaluate(num_episodes_in_eval, cfg['num_steps_in_eval'], False, make_graphs=True)

            graph_dirname = os.path.join(os.path.dirname(cfg["initial_params_filename"]), "graph")
            os.makedirs(graph_dirname, exist_ok=True)

            # variables for plot
            episodes = []
            rewards = []
            num_failures = []
            for i in range(num_episodes_in_eval):
                episodes.append(i + 1)
            for i in range(num_episodes_in_eval):
                rewards.append(r[i][0])
                num_failures.append(len(r[i][1])) if len(r[i][1]) <= 4 else num_failures.append(4) # regard more than 4 as 4

            # scatter diagram with colorbar
            plt.scatter(episodes, rewards, s=30, c=num_failures, cmap="binary", lw=0.5, edgecolors="k", vmin=0, vmax=4)
            plt.hlines(sum(rewards) / len(rewards), 0, num_episodes_in_eval, colors="r")
            plt.colorbar(label="num_failures")
            plt.xlim(0, num_episodes_in_eval)
            plt.ylim(-100, 1300)
            plt.xlabel("episode")
            plt.ylabel("reward")
            filename = os.path.join(graph_dirname, f"scatter_diagram_with_colorbar_{num_episodes_in_eval}_episodes.png")
            plt.savefig(filename)

            # scatter diagram
            plt.scatter(episodes, rewards)
            plt.hlines(sum(rewards) / len(rewards), 0, num_episodes_in_eval, colors="r")
            plt.xlim(0, num_episodes_in_eval)
            plt.ylim(-100, 1300)
            plt.xlabel("episode")
            plt.ylabel("reward")
            filename = os.path.join(graph_dirname, f"scatter_diagram_{num_episodes_in_eval}_episodes.png")
            plt.savefig(filename)

            # histogram
            plt.hist(rewards, range=(-100, 1300), rwidth=0.9, bins=28, orientation="horizontal")
            plt.hlines(sum(rewards) / len(rewards), 0, num_episodes_in_eval, colors="r")
            plt.xlim(0, num_episodes_in_eval / 2)
            plt.ylim(-100, 1300)
            plt.ylabel("reward")
            filename = os.path.join(graph_dirname, f"histogram_{num_episodes_in_eval}_episodes.png")
            plt.savefig(filename)

            # violinplot
            plt.violinplot(rewards, showmeans=True)
            plt.ylim(-100, 1300)
            plt.ylabel("reward")
            filename = os.path.join(graph_dirname, f" violinplot_{num_episodes_in_eval}_episodes.png")
            plt.savefig(filename)

            break

    elif args.type == "record":
        # Create a window to init GLFW.
        GlfwContext(offscreen=True)
        # print(f"camera: {model.env.model._camera_name2id}")

        num_steps = cfg["num_steps_in_eval"]
        num_episodes = 1
        size = (640, 480)
        # size = (1280, 960)
        frame_rate = 30
        if cfg_fullname is not None:
            videoname = os.path.basename(os.path.dirname(cfg_fullname))
        else:
            videoname = os.path.basename(cfg['initial_params_filename'])
        video_path = f"./log/video/{videoname}.mp4"
        print(f"path: {video_path}")

        scale = size[0] / 640
        fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(video_path, fmt, frame_rate, size)
        for i in range(num_episodes):
            obs = model.env.reset()
            for t in range(num_steps + 1):
                rgb_array = model.env.render(mode="rgb_array", width=size[0], height=size[1])
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr_array, f"step: {t}", (int(5 * scale), int(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8 * scale, (255, 255, 255), thickness=2)
                # cv2.putText(bgr_array, f"episode: {i+1}, step: {t}", (int(5 * scale), int(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.8 * scale, (255, 255, 255), thickness=2)
                writer.write(bgr_array)

                act = model.get_action(obs)
                obs, reward, done, info = model.env.step(act)
        print("record done")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
