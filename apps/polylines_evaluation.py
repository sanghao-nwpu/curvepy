from curvepy.core import *

def find_associated_polyline(polyline_est: Polyline, candidate_polylines: list[Polyline]) -> Polyline:
    for candidata_polyline in candidate_polylines:
        error = polyline_est.calculate_distance_to_another_polyline(candidata_polyline)
        
def evaluate_polylines(polylines_est: list[Polyline], polylines_gt: list[Polyline]):
    


def main() -> None:
    pass


if __name__ == '__main__':
    main()
