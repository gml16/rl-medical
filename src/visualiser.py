import argparse
from medical import MedicalPlayer

IMAGE_SIZE = (45, 45, 45)

def main(files_list, file_type, landmark_ids, agents):
    env = MedicalPlayer(
        screen_dims=IMAGE_SIZE,
        viz=0.01,
        saveGif=False,
        saveVideo=False,
        task="eval",
        files_list=files_list,
        file_type=file_type,
        landmark_ids=landmark_ids,
        history_length=1,
        multiscale=False,
        agents=agents
    )
    seen_files = set()
    while True:
        env.reset(fixed_spawn = "on_landmark")
        file_name = env._image[0].name
        if file_name in seen_files:
            break
        seen_files.add(file_name)
        print(f"Dimension of image {file_name}: {env._image[0].dims}")
        print("Landmarks", tuple(map(tuple, env._target_loc)))
        env.display()
        print("Press Enter to go to the next image...")
        input()
    print("All images have been visualised.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--file_type', help='Type of the training and validation files',
        choices=['brain', 'cardiac', 'fetal'], default='train')
    parser.add_argument(
        '--files', type=argparse.FileType('r'), nargs='+',
        help="""Filepath to the text file that contains list of images.
                Each line of this file is a full path to an image scan.
                For (task == train or eval) there should be two input files
                ['images', 'landmarks']""")
    parser.add_argument(
        '--landmarks', nargs='*', help='Landmarks to use in the images',
        type=int, default=[1])
    parser.set_defaults(write=False)
    args = parser.parse_args()

    agents = len(args.landmarks)
    main(args.files, args.file_type, args.landmarks, agents)