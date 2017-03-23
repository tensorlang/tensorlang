import { join } from 'path';
import { dialog } from 'electron';
import { writeFileObservable, createSymlinkObservable } from '../utils/fs';

const fs = require('fs');

const getStartCommand = () => {
  const subdir = (process.platform === 'darwin') ? 'MacOS' : '';
  const dir = join(process.resourcesPath, '..', subdir);

  const naoPlayPath = join(dir, `nao-play${ext}`);

  if (fs.existsSync(naoPlayPath)) {
    return [naoPlayPath, '', join(process.resourcesPath, 'bin', '')];
  }
  return [null, null, null];
};

const installShellCommandsObservable = (exe, rootDir, binDir) => {
  const envFile = join(binDir, 'nao-play-env');
  return writeFileObservable(envFile, `NAO_PLAY_EXE="${exe}"\nNAO_PLAY_DIR="${rootDir}"`)
    .flatMap(() => {
      const target = join(binDir, 'nao-play.sh');
      return createSymlinkObservable(target, '/usr/local/bin/nao-play')
        .catch(() => {
          const dest = join(process.env.HOME, '.local/bin/nao-play');
          return createSymlinkObservable(target, dest);
        });
    });
};

export const installShellCommand = () => {
  const [exe, rootDir, binDir] = getStartCommand();
  if (!exe) {
    dialog.showErrorBox(
      'nao-play application not found.',
      'Could not locate nao-play executable.'
    );
    return;
  }

  installShellCommandsObservable(exe, rootDir, binDir)
    .subscribe(
      () => {},
      err => dialog.showErrorBox('Could not write shell script.', err.message),
      () => dialog.showMessageBox({
        title: 'Command installed.',
        message: 'The shell command "nao-play" is installed.',
        detail: 'Get help with "nao-play --help".',
        buttons: ['OK']
      })
    );
};
