# Changelog

## [0.6.3](https://github.com/manishsharma0398/documind/compare/v0.6.2...v0.6.3) (2026-08-23)


### Bug Fixes

* pair embeddings on the API's index, not on position ([#34](https://github.com/manishsharma0398/documind/issues/34)) ([2a4b666](https://github.com/manishsharma0398/documind/commit/2a4b6662dd5e2528ca0cca7339f622e554007f3d))

## [0.6.2](https://github.com/manishsharma0398/documind/compare/v0.6.1...v0.6.2) (2026-08-23)


### Bug Fixes

* keep the cost record when an ingest run dies ([#31](https://github.com/manishsharma0398/documind/issues/31)) ([c12dd67](https://github.com/manishsharma0398/documind/commit/c12dd67debb87bca715d4d65ff4f6959aef22c70))

## [0.6.1](https://github.com/manishsharma0398/documind/compare/v0.6.0...v0.6.1) (2026-08-22)


### Documentation

* bring the readme up to date with what ships ([#28](https://github.com/manishsharma0398/documind/issues/28)) ([6c12614](https://github.com/manishsharma0398/documind/commit/6c126140569fe8a580d97f1d9182ac26e8e5b190))

## [0.6.0](https://github.com/manishsharma0398/documind/compare/v0.5.0...v0.6.0) (2026-08-22)


### Features

* let a request exclude sources by glob ([#24](https://github.com/manishsharma0398/documind/issues/24)) ([4e3a9d8](https://github.com/manishsharma0398/documind/commit/4e3a9d85999e9adc679b50dcdc953aa40790439d))
* log every request with a correlation id ([#26](https://github.com/manishsharma0398/documind/issues/26)) ([266353b](https://github.com/manishsharma0398/documind/commit/266353b8e55456bbdc6d9270a1cafdebba86093f))
* skip unchanged files on re-ingest ([#23](https://github.com/manishsharma0398/documind/issues/23)) ([a30366f](https://github.com/manishsharma0398/documind/commit/a30366f7c3395a57d879699bd570131427de8dc2))

## [0.5.0](https://github.com/manishsharma0398/documind/compare/v0.4.0...v0.5.0) (2026-08-21)


### Features

* add openai client and map its failures to status codes ([#21](https://github.com/manishsharma0398/documind/issues/21)) ([7d18b70](https://github.com/manishsharma0398/documind/commit/7d18b7060af9f60bdd809e87fb3d6580e46d1d02))

## [0.4.0](https://github.com/manishsharma0398/documind/compare/v0.3.0...v0.4.0) (2026-08-21)


### Features

* header-aware chunking with section breadcrumbs ([#17](https://github.com/manishsharma0398/documind/issues/17)) ([077d860](https://github.com/manishsharma0398/documind/commit/077d86093717d68fd528bd18c90f18ab22839189))


### Bug Fixes

* reserve breadcrumb tokens so TOKEN_SIZE is a real ceiling ([#19](https://github.com/manishsharma0398/documind/issues/19)) ([e2dd3cf](https://github.com/manishsharma0398/documind/commit/e2dd3cff30300d1c19df9b915837e995217d788b))
* stop repeating section headings inside the chunk ([#20](https://github.com/manishsharma0398/documind/issues/20)) ([fffd9e1](https://github.com/manishsharma0398/documind/commit/fffd9e15a74a86d884fcc33740d3d04ab5f52d67))

## [0.3.0](https://github.com/manishsharma0398/documind/compare/v0.2.0...v0.3.0) (2026-08-20)


### Features

* add filesystem document source with settings and error handling ([#15](https://github.com/manishsharma0398/documind/issues/15)) ([c2133aa](https://github.com/manishsharma0398/documind/commit/c2133aa213079cb6f5748499ec3389d00031a100))

## [0.2.0](https://github.com/manishsharma0398/documind/compare/v0.1.2...v0.2.0) (2026-08-20)


### Features

* add async qdrant client with lifespan wiring ([#13](https://github.com/manishsharma0398/documind/issues/13)) ([0c56209](https://github.com/manishsharma0398/documind/commit/0c5620990faa41ba6237d456998894308dd0e1de))

## [0.1.2](https://github.com/manishsharma0398/documind/compare/v0.1.1...v0.1.2) (2026-08-19)


### Bug Fixes

* sync uv.lock on release and sanitise branch slugs ([#11](https://github.com/manishsharma0398/documind/issues/11)) ([14cbd6a](https://github.com/manishsharma0398/documind/commit/14cbd6a40a797e41377f946b5009031e8d60c9ee))

## [0.1.1](https://github.com/manishsharma0398/documind/compare/v0.1.0...v0.1.1) (2026-08-19)


### Bug Fixes

* bump pyproject version on release ([#9](https://github.com/manishsharma0398/documind/issues/9)) ([f811956](https://github.com/manishsharma0398/documind/commit/f8119560d382d08ba8bd552bcccc4c0e98489e8c))

## 0.1.0 (2026-08-19)


### Bug Fixes

* replace placeholder package description ([#7](https://github.com/manishsharma0398/documind/issues/7)) ([9c40125](https://github.com/manishsharma0398/documind/commit/9c40125e26d1f87227d0ed460dc2a9086bba5469))
