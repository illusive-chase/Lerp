# Lerp

+ Minimal implementation of piecewise linear interpolation in arbitrary dimensions. (with CGAL & Eigen)

### Usage

See `main.cpp` .

```cpp
int main()
{
	std::vector<std::array<double, 2>> points = {
		{0,0},
		{0,1},
		{1,0}
	};

	Lerp lerp{ points };

	if (!lerp.is_valid()) return 1;

	lerp.dump(std::cout);
	std::cout << std::endl;

	auto result = lerp.interp(std::array<double, 2>{ 0.5, 0.5 });
	if (result == std::nullopt) return 1;
	auto&& [idxs, weights] = *result;

	for (int idx : idxs) {
		std::cout << idx << ' ';
	}
	std::cout << std::endl;

	for (double weight : weights) {
		std::cout << weight << ' ';
	}
	std::cout << std::endl;

	return 0;
}
```

