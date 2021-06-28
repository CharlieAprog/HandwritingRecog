import numpy as np
from heapq import *
from tqdm import tqdm


def heuristic(a, b):
    return (b[0] - a[0])**2 + (b[1] - a[1])**2


def astar(array, start, goal, i):
    # 8 directions: up, down, right, left, ....
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1),
                 (-1, -1)]
    close_set = set()
    came_from = {}  # prev step
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:
        current = heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if i == 11:
                # print(current)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:  # array bound y walls
                    continue
            else:  # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(
                    neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [
                    i[1] for i in oheap
            ]:
                if i == 11:
                    # print(current)
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + \
                    heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
    return []


def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)


def path_exists(window_image):
    # very basic check first then proceed to A* check
    if 0 in horizontal_projections(window_image):
        return True

    padded_window = np.zeros((window_image.shape[0], 1))
    world_map = np.hstack(
        (padded_window, np.hstack((window_image, padded_window))))
    path = np.array(
        astar(world_map, (int(world_map.shape[0] / 2), 0),
              (int(world_map.shape[0] / 2), world_map.shape[1]), 0))
    if len(path) > 0:
        return True

    return False


def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False

    for col in range(nmap.shape[1]):
        start = col
        end = col + 40
        if end > nmap.shape[1] - 1:
            end = nmap.shape[1] - 1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
            road_blocks.append(col)

        if needtobreak == True:
            break

    return road_blocks


def group_the_road_blocks(road_blocks):
    # group the road blocks
    road_blocks_cluster_groups = []
    road_blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        road_blocks_cluster.append(value)
        # split up the clusters
        if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1 or \
           index == size-1 and len(road_blocks_cluster) > 0:
            road_blocks_cluster_groups.append(
                [road_blocks_cluster[0], road_blocks_cluster[-1]])
            road_blocks_cluster = []
    return road_blocks_cluster_groups


def find_paths(hpp_clusters, binary_image, avg_lh):
    fake_rb_indices = []
    agent_height = []
    upward_push = int(avg_lh * 0.85)
    for idx, cluster_of_interest in enumerate(hpp_clusters):
        # print(idx)
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[-1]]
        road_blocks = get_road_block_regions(nmap)
        start_end_height = int(nmap.shape[0] / 2)
        agent_height.append(start_end_height)

        # check for fake roadblocks
        if len(road_blocks) != 0:
            nmap_rb = binary_image[cluster_of_interest[0] -
                                   upward_push:cluster_of_interest[-1]]
            road_blocks_new = get_road_block_regions(nmap_rb)
            if road_blocks_new != road_blocks and len(road_blocks_new) < len(
                    road_blocks):
                # print('Fake roadblock has been hit, better path found')
                fake_rb_indices.append(idx)
                road_blocks = road_blocks_new
        road_blocks_cluster_groups = group_the_road_blocks(road_blocks)

        # create the doorways for real roadblocks
        for road_blocks in road_blocks_cluster_groups:
            rb_end_reached = False  # true end of the roadblock
            i = 0
            prev_pixel = binary_image[
                cluster_of_interest[0]:
                cluster_of_interest[-1], :][:, road_blocks[0]:binary_image.
                                            shape[1] - 1][0, 0]
            # making sure prev_pixel is initiated with a 0
            step_back = 1
            while prev_pixel:
                prev_pixel = binary_image[
                    cluster_of_interest[0]:
                    cluster_of_interest[-1], :][:, road_blocks[0] -
                                                step_back:binary_image.
                                                shape[1] - 1][0, 0]
                step_back += 1
            assert prev_pixel == 0, "prev_pixel=1 at the start of annulling, horizontal cut cannot be performed"

            while True:
                i += 1
                if binary_image[cluster_of_interest[0]:cluster_of_interest[
                        -1], :][:, road_blocks[0]:binary_image.shape[1] -
                                1][0, i] == 0:
                    if prev_pixel == 1:
                        rb_end_reached = True
                        binary_image[
                            cluster_of_interest[0]:cluster_of_interest[
                                -1], :][:,
                                        road_blocks[0]:binary_image.shape[1] -
                                        1][0, 0:i] = 0
                    if rb_end_reached:
                        # detect fake roadblock end
                        fake_end_length = 20
                        if len(
                                np.nonzero(
                                    binary_image[cluster_of_interest[0]:
                                                 cluster_of_interest[-1], :]
                                    [:, road_blocks[0]:binary_image.shape[1]][
                                        0, i:i + fake_end_length])[0]) != 0:
                            rb_end_reached = False
                            prev_pixel = 0
                            # print("fake end")
                            continue
                        # true end
                        break
                    prev_pixel = 0
                else:
                    prev_pixel = 1

            # Plot enlargened section (if needed) where horizontal cut is performed
            # if idx == 6:
            #     plt.plot(figsize=(16, 12))
            #     plt.imshow(invert(binary_image[cluster_of_interest[0]:cluster_of_interest[-1],:][:, road_blocks[0]: road_blocks[1]]), cmap="gray")
            #     plt.show()

    paths = []
    for i, cluster_of_interest in tqdm(enumerate(hpp_clusters)):
        if i in fake_rb_indices:
            nmap = binary_image[cluster_of_interest[0] -
                                upward_push:cluster_of_interest[-1]]
            offset_from_top = cluster_of_interest[0] - upward_push
            height = agent_height[i] + upward_push
        else:
            nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[-1]]
            offset_from_top = cluster_of_interest[0]
            height = agent_height[i]
        path = np.array(
            astar(nmap, (height, 0), (height, nmap.shape[1] - 1), i))
        if path.shape[0] == 0:
            continue
        path[:, 0] += offset_from_top
        path = [list(step) for step in path]
        paths.append(path)
    return paths