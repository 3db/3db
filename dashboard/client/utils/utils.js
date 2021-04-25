export function argMax(array) {
    return [].reduce.call(array, (m, c, i, arr) => c > arr[m] ? i : m, 0)
}
export function  nn(array, value) {
  const distances = array.map(x => Math.abs(value - x));
  const best_distance = min(distances);
  return distances.indexOf(best_distance);
}