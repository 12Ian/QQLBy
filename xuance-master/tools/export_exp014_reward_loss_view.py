"""Derive a small reward/loss TensorBoard view from the frozen 014 event stream."""
import hashlib
import json
import mmap
from pathlib import Path
import struct

from torch.utils.tensorboard import SummaryWriter
from xuance.engine.uav_baseline_logging import RewardLossTensorBoard

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/reviews/exp014_reward_loss_view_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def varint(data, pos):
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7


def main():
    assert not OUT.exists(), 'Preserve existing derived view'
    frozen = ROOT/'outputs/acceptance/exp014_baseline_v1'
    reference = json.loads((frozen/'tensorboard_archive.json').read_text())['entries']
    assert len(reference) == 1
    source = ROOT/reference[0]['source']
    assert sha(source) == reference[0]['sha256']
    OUT.mkdir(parents=True)
    logdir = OUT/'logs/seed_derived'
    writer = SummaryWriter(str(logdir))
    compact = RewardLossTensorBoard(writer)
    episode, episode_step, episodes, records = {}, 0, 0, 0
    with source.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        pos = 0
        while pos < len(data):
            length = struct.unpack_from('<Q', data, pos)[0]
            p, end = pos+12, pos+12+length
            pos = end+4
            assert pos <= len(data) and data[p] == 9
            p += 9
            step = 0
            if p < end and data[p] == 16:
                step, p = varint(data, p+1)
            if p == end or data[p] != 42:
                continue
            size, p = varint(data, p+1)
            assert p+size == end and data[p] == 10
            size, p = varint(data, p+1)
            assert p+size == end and data[p] == 10
            size, p = varint(data, p+1)
            tag = data[p:p+size]
            p += size
            assert data[p] == 21 and p+5 == end
            records += 1
            if tag == b'Train-Episode/Completed-Episodes':
                if episode:
                    compact.log(episode, episode_step)
                    episodes += 1
                episode, episode_step = {}, step
            elif (tag.startswith((b'Train-Returns/', b'Train-Rewards/')) or
                  tag == b'Train-Curriculum/Level'):
                episode[tag.decode()] = struct.unpack_from('<f', data, p+1)[0]
            elif tag in (b'Train-Learning/loss_actor_mean', b'Train-Learning/loss_critic_mean') or tag.startswith(b'Test-Results/Episode-Rewards-Level-'):
                compact.log({tag.decode(): struct.unpack_from('<f', data, p+1)[0]}, step)
            if records % 2000000 == 0:
                print('READ_SCALARS', records, flush=True)
    if episode:
        compact.log(episode, episode_step)
        episodes += 1
    compact.flush_rewards()
    writer.close()
    assert episodes == 73169
    files = [{'path':str(p.relative_to(OUT)), 'bytes':p.stat().st_size, 'sha256':sha(p)}
             for p in logdir.glob('events.out.tfevents*')]
    result = dict(status='completed', derived_from_experiment='014', source=reference[0],
        original_scalar_records=records, original_episodes=episodes,
        reward_batch_episodes=20, losses='Existing actual-update CSV window means (up to 1000 updates)',
        step_axis='Original environment transitions', wall_time_axis='Export timestamps',
        model_sha256=json.loads((frozen/'manifest.json').read_text())['model_sha256'],
        code_sha256={rel:sha(ROOT/rel) for rel in ['tools/export_exp014_reward_loss_view.py',
                           'xuance/engine/uav_baseline_logging.py']},
        files=files, training_started=False, frozen_baseline_modified=False)
    (OUT/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
