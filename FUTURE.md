신경 쓸 수 있지만 안 해도 되는 것들



&nbsp; 1. train\_shakespeare\_vanilla.py에 새 플래그 미반영 —

&nbsp; --use-cosine-schedule, --warmup-steps 등이 NGT 학습 스크립트에만

&nbsp; 추가됨. Vanilla 쪽에서도 동일 조건으로 비교하려면 추가해야 하지만,

&nbsp; 직접 복사해서 쓰면 됨

&nbsp; 2. train\_shakespeare.py에 build\_vocab/encode 잔존 — probe words 기능

&nbsp; 때문에 아직 남아있음. tokenizer\_utils로 완전 통합 가능하지만, probe

&nbsp; words가 char tokenizer 전용이라 실익이 크지 않음

&nbsp; 3. 루트에 nul 파일 — Windows에서 생긴 빈 파일로 보임. .gitignore에

&nbsp; 추가하거나 삭제하면 깔끔함



&nbsp; 나머지는 전부 실험 영역이 맞습니다:



&nbsp; - gravity bias가 실제로 loss를 줄이는지

&nbsp; - weight tying이 이 규모에서 효과가 있는지

&nbsp; - cosine schedule + warmup 최적 조합

&nbsp; - repulsion interval을 늘렸을 때 좌표 collapse가 안 생기는지

&nbsp; - 기존 5k 체크포인트와 새 아키텍처 성능 비교

